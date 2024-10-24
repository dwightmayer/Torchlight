#!/usr/bin/env python

import torch
from torch import nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
import regex as re


import time

## Get data

ds = load_dataset("wikipedia", "20220301.simple", split='train', trust_remote_code=True) # 235MB subset of wikipedia
print('dataset loaded')

# ds0 = load_dataset("openbmb/UltraInteract_sft", split='train', trust_remote_code=True) # 151 MB of  code (finetune)
# ds1 = load_dataset("wikipedia", "20220301.en", streaming=True) # 21GB of English Wikipedia
# ds2 = load_dataset("pythera/english-mlmcorpus", streaming=True) # 58GB of plain text
# ds3 = load_dataset("H-D-T/Buzz-slice-1-10-V1.2", streaming=True) # 2.5GB of code related
# ds4 = load_dataset("nvidia/OpenMathInstruct-1", streaming=True) # 2.7GB of Math instruct

# Concatenate datasets // hard to combine between formats
dataset_cc = concatenate_datasets([ds])
print('datasets concatenated')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=5, return_tensors='pt')


print('Beginning tokenization:')
tokenized_datasets = dataset_cc.map(tokenize_function, batched=True)
print('tokenization mapped')

# dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)

print('creating train dataset...')
train_dataset = (Dataset.from_dict({'input_ids': [x['input_ids'] for x in tokenized_datasets],
                                   'labels': [x['input_ids'] for x in tokenized_datasets]})) ### USED TO BE TITLE
print('training dataset object created')


# Model Config class for Hugging Face compatibility
class TransformerLMConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size=tokenizer.vocab_size,
                 embedding_dim=64,
                 hidden_dim=128,
                 n_heads=8,
                 num_layers=4,
                 sequence_length=5,
                 **kwargs):

        super().__init__(**kwargs)
        # Set defaults if not provided

        # // Set to <100 and then run through small sample text to ensure system catches errors
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length


# Define the Transformer Model using Hugging Face's PreTrainedModel
class TransformerDecoderLM(PreTrainedModel):
    config_class = TransformerLMConfig

    def __init__(self, config):
        super().__init__(config)

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional encoding (alternative)
        self.register_buffer("pos_encoding",
                             self.generate_positional_encoding(config.embedding_dim, config.sequence_length))

        # Transformer layers
        decoder_transformer_layer = nn.TransformerDecoderLayer(d_model=config.embedding_dim,
                                                               nhead=config.n_heads,
                                                               dim_feedforward=config.hidden_dim,
                                                               batch_first=True)

        self.transformer = nn.TransformerDecoder(decoder_transformer_layer, num_layers=config.num_layers)

        # Final linear layer for word prediction
        self.fc = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, input_ids, labels=None):

        # Checks for vocab size, I'd like to reset the config vocab size
        assert input_ids.max().item() < self.config.vocab_size, f"input_ids contain indices out of range: {input_ids.max().item()} >= {self.config.vocab_size}"
        if self.config.vocab_size < input_ids.max().item():
            # This resets the vocab size to match the input ids, this might explode something...
            self.config.vocab_size = input_ids.max().item()

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # embeds = self.embedding(input_ids) + self.pos_encoding[:input_ids.size(1), :]
        embeds = self.embedding(input_ids) + self.pos_encoding[:seq_len, :]

        # Create causal mask (upper triangular to prevent attending to future tokens)
        # tgt_mask = self.generate_square_subsequent_mask(input_ids.size(1))
        tgt_mask = self.generate_square_subsequent_mask(seq_len)

        # Gets transformer outputs
        transformer_out = self.transformer(embeds, memory=embeds, tgt_mask=tgt_mask)
        # logits = self.fc(transformer_out[:, -1, :]) # Predicts the last token nin the sequence...
        logits = self.fc(transformer_out)  # Predicts for all tokens in the sequence

        loss = None
        if labels is not None:

            # Reshape tensors
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)

            # This is the loss function!!!!
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    # This should not be a static method. PyCharm is mistaken imo.
    def generate_positional_encoding(self, dim, max_len):
        # This gets the positional encoding
        pos_enc = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim))

        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        # Create an upper triangular matrix of -inf (for masking future positions)
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


config = TransformerLMConfig()
model = TransformerDecoderLM(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',     # Output directory
    num_train_epochs=100,       # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device // usually (256)
    logging_dir='./logs',       # Directory for logs
    logging_steps=10,
    save_steps=1000,
    save_total_limit=3,
    use_cpu=False,
    fp16=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
print('beginning model training loop')
trainer.train()
print(f'Vocab Size: {config.vocab_size}')
print('model training loop complete :)')

