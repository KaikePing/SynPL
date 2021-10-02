import nltk, torch, operator
import pandas as pd
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from itertools import combinations
from numpy import random, logical_not
from datasets import DatasetDict, Dataset
from transformers import DataCollatorWithPadding, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments
from functools import reduce
from math import ceil

# Download WordNet
nltk.download('wordnet')

# global para
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
N = 5

# Get through the words in the tokenizer
synonyms = []
for synset in list(wn.all_synsets()):
  temp = synset.lemma_names()[:]
  if len(temp)<=1:
    continue
  temp2 = []
  for i in temp:
    if len(tokenizer(i)['input_ids'])==3:
      temp2.append(i)
  if len(temp2)<=1:
    continue  
  synonyms.append(temp2)

# Extract words
df = []
for line in synonyms:
  comb = combinations(line, 2)
  for i in list(comb):
    temp1 = line[:]
    temp1.remove(i[0])
    temp2 = line[:]
    temp2.remove(i[1])
    df.append({
      'word1': i[0],
      'word2': i[1],
      'synonyms1': temp1,
      'synonyms2': temp2,
    })

# build sentences
datasets = []
for line in df:
  sen = f'{line["word1"]} is close in meaning to {line["word2"]}.'
  datasets.append({'sen':sen, 'word1':line["word1"], 'word2':line["word2"], 'synonyms':line['synonyms1']})

# Creat datasets
random.seed(123)
## Random pick some words as test data
datasets_df = pd.DataFrame(datasets)
temp = datasets_df['word1'].unique()
test_words = random.choice(temp, replace=False, size=round(0.3*temp.size))
## Extract those words
flag = datasets_df['word1'].isin(test_words)
test = datasets_df.loc[flag,:].reset_index(drop=True)
train = datasets_df.loc[logical_not(flag),:].reset_index(drop=True)
## Clear test dataset, word1 can only appear once
test = test.groupby('word1').agg({'sen':'first', 'synonyms': sum, 'word2':'first'}).reset_index()
def get_syn(row):
  synonyms = []
  for syn in wn.synsets(row):
    for l in syn.lemmas():
      if len(tokenizer(l.name())['input_ids'])==3:
        synonyms.append(l.name())
  synonyms = list(set(synonyms))
  synonyms.remove(row)
  return synonyms
test['synonyms'] = test['word1'].apply(lambda x: get_syn(x))

# Load into Dataset
datasets = DatasetDict({
    'train': Dataset.from_pandas(train),
    'test': Dataset.from_pandas(test)})

# brute-force approach O(n*m)
def findindex(seq, subseq):
  # Usage: findindex([4,3,'a',5,6], [5,6])
  i, n, m = -1, len(seq), len(subseq)
  try:
    while True:
      i = seq.index(subseq[0], i + 1, n - m + 1)
      if subseq == seq[i:i + m]:
        return i
  except ValueError:
    return -1

# Custom tokenize
def tokenize_function(examples):
  if length==0:
    sen_no_mask = tokenizer(examples["sen"], truncation=True, padding=True)
  else:
    sen_no_mask = tokenizer(examples["sen"], truncation=True, padding='max_length', max_length=length)
  masked_word = tokenizer(examples["word2"], truncation=True, padding=True)['input_ids']
  inputs = []
  labels = []
  
  for i in range(len(masked_word)):
    # Find the word encoding part
    start = masked_word[i].index(101)
    end = masked_word[i].index(102)
    # Extract this part
    temp = [masked_word[i][j] for j in range(start+1,end)]
    # Find the same part in input and mask them
    ipt = sen_no_mask['input_ids'][i][:]
    idx = findindex(ipt, temp)
    for j in range(len(temp)):
      ipt[idx+j] = tokenizer.mask_token_id
    inputs.append(ipt)
    # Find the other part and replace those unmasked indices with -100
    label = sen_no_mask['input_ids'][i][:]
    for j in range(len(ipt)):
      if ipt[j]!=tokenizer.mask_token_id:
        label[j]=-100
    labels.append(label)
  
  # encode synonyms
  synonyms = []
  for i in examples["synonyms"]:
    temp = tokenizer(i)['input_ids']
    synonym = []
    for j in temp:
      synonym.append(j[1])
    synonyms.append(synonym)

  sen_no_mask['input_ids']=inputs
  sen_no_mask['label']=labels
  sen_no_mask['synonyms']=synonyms
  return sen_no_mask

length = 0
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["sen", "word2", "word1"])

# Data collator
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load base model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Check the performance of original PM
# Accurcy is defined as if the true answer is in its top N predictions.
def common_member(a, b):
  a_set = set(a)
  b_set = set(b)
  if (a_set & b_set):
    return True 
  else:
    return False

def check_accuracy(df, lb, outputs, n=5):
  mask_token_index = torch.where(df['input_ids'] == tokenizer.mask_token_id)[1]
  mask_token_logits = outputs.logits[range(outputs.logits.size()[0]), mask_token_index, :]
  top_n_tokens = torch.topk(mask_token_logits, n, dim=1).indices.tolist()
  return [1 if common_member(i, top_n_tokens[j]) else 0 for j,i in enumerate(lb)]

def get_accuracy(df, shards):
  accuracies = []
  for i in range(shards):
    pm_inputs = df.shard(num_shards=shards, index=i)
    test_inputs = dict((k, torch.LongTensor(pm_inputs[k])) for k in ['attention_mask', 'input_ids', 'token_type_ids'])
    test_labels = torch.LongTensor(pm_inputs['label'])
    outputs = model(**test_inputs, labels=test_labels)
    accuracy = check_accuracy(test_inputs, pm_inputs['synonyms'], outputs, N)
    accuracies.append(accuracy)
  return reduce(operator.concat, accuracies)

# Accuracy for raw BERT
accuracy = get_accuracy(tokenized_datasets['test'], 20)
print(f'Accuracy for raw BERT is: {sum(accuracy)/len(accuracy)}')

# Trainer
batch_size=16
logging_steps=ceil(tokenized_datasets['train'].num_rows/batch_size)
num_train_epochs=1

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy = "epoch",
    learning_rate=2e-6,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_steps=3000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=collator
)

trainer.train()

# Calculate the accuracy
temp = trainer.predict(tokenized_datasets["test"])

def cal_accuracy(input_ids, lb, outputs, n=5):
  mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
  mask_token_logits = outputs[range(outputs.size()[0]), mask_token_index, :]
  top_n_tokens = torch.topk(mask_token_logits, n, dim=1).indices.tolist()
  return [1 if common_member(i, top_n_tokens[j]) else 0 for j,i in enumerate(lb)]

accuracy = cal_accuracy(torch.Tensor(tokenized_datasets["test"]["input_ids"]), tokenized_datasets["test"]['synonyms'], torch.Tensor(temp.predictions), N)
print(f'Accuracy after finetune is: {sum(accuracy)/len(accuracy)}')