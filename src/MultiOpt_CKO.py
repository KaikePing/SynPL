import torch
import pandas as pd
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from dataclasses import dataclass
from typing import Optional, Union

# Global Para
model_checkpoint = "bert-base-uncased"
batch_size = 16

# Load the TOEFL data
df = pd.read_csv('Vocab440.csv', index_col='ID')
df.Ans = [ ord(x) - 65 for x in df.Ans ]
## Extracting the keyword in the question
df['Keya'] = df['Question'].str.extract(r'((?<=").*(?="))')
df['Keya'][258] = 'counterpart'

# Split the data into train and test data
random.seed(123)
train, test = train_test_split(df, test_size=0.3)

# Load into Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train),
    'test': Dataset.from_pandas(test)})

dataset = dataset.rename_column("Ans", "label")

# init Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Define Tokenizer method
var_names = ["Opt1", "Opt2", "Opt3", "Opt4"]
def Tokenize(df):
  first = [[context]*4 for context in df['Keya']]
  second = [[context]*4 for context in df['Context']]
  third = [[df[i][j] for i in var_names] for j in range(len(df["Context"]))]
  
  first2 = sum(first, []) # keywords
  second2 = sum(second, []) # context
  third2 = sum(third, []) # options

  tokenized = tokenizer([f"{second2[i]} {tokenizer.sep_token} {first2[i]} {tokenizer.sep_token} {third2[i]}" for i in range(len(first2))], truncation=True, add_special_tokens=True)
  # Un-flatten
  return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}

# Tokenize
encoded_datasets = dataset.map(Tokenize, batched=True, batch_size=20)

# Load raw model
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

# Set parameters for Trainer
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=12,
    weight_decay=0.01,
    seed=123,
    logging_steps=10
)

# Custom data_collator_for_multiple_choice
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

# Calculate the accuracy during training
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

# Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()