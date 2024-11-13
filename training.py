#!/usr/bin/env python

import torch
from torch import nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np

import time

# Get data
ds0 = load_dataset("openbmb/UltraInteract_sft", split='train', streaming=True, trust_remote_code=True) # 151 MB of  code (finetune)
ds1 = load_dataset("wikipedia", "20220301.en", split='train', streaming=True, trust_remote_code=True) # 21GB of English Wikipedia // IterableDatasetDict 42
ds2 = load_dataset("pythera/english-mlmcorpus", split='train', streaming=True, trust_remote_code=True) # 58GB of plain text // IterableDatasetDict 100
ds3 = load_dataset("H-D-T/Buzz-slice-1-10-V1.2", split='train', streaming=True, trust_remote_code=True) # 2.5GB of code related // IterableDatasetDict 1
ds4 = load_dataset("nvidia/OpenMathInstruct-1", streaming=True, trust_remote_code=True) # 2.7GB of Math instruct // Iterable Dataset Dict 2
ds5 = load_dataset('bookcorpus', split='train', streaming=True, trust_remote_code=True) # ??? GB of text

# It is possible to rename columns for better text combination...
ds1 = ds1.remove_columns(['id', 'url', 'title'])
datasets = [ds1, ds2, ds5]
dataset = concatenate_datasets(datasets)

# Creates Dataloader using only text column /// ITERABLE
dl = DataLoader(dataset, batch_size=100_000, shuffle=None, batch_sampler=None, sampler=None)
print('Dataloader created')


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=5, return_tensors='pt')


# Model Config class for Hugging Face compatibility
class TransformerLMConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size=tokenizer.vocab_size,
                 embedding_dim=256, # increasing embedding dimension rq
                 hidden_dim=128,
                 n_heads=16,
                 num_layers=16,
                 sequence_length=5,
                 **kwargs):

        super().__init__(**kwargs)
        # Set defaults if not provided

        # Sets vocab size
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Pythia 1B: n_layers = 16, d_model / embedding_dim(?) = 2048, n_heads=8, d_heads=256, batch_sizze=2M
        # Do I want to 4x n_layers and 32x embedding dimension...


# Define the Transformer Model using Hugging Face's PreTrainedModel
class TransformerDecoderLM(PreTrainedModel):
    config_class = TransformerLMConfig

    def __init__(self, config):
        super().__init__(config)

        # Creates embedding layer, positional encodings
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.register_buffer("pos_encoding",
                             self.generate_positional_encoding(config.embedding_dim, config.sequence_length))

        # Transformer layers
        decoder_transformer_layer = nn.TransformerDecoderLayer(d_model=config.embedding_dim,
                                                               nhead=config.n_heads,
                                                               dim_feedforward=config.hidden_dim,
                                                               batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_transformer_layer, num_layers=config.num_layers)

        # Final linear layer for word prediction.py
        self.fc = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, input_ids, labels=None):

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Creates embeddings with positional encodings
        embeds = self.embedding(input_ids) + self.pos_encoding[:seq_len, :]

        # Create causal mask (upper triangular to prevent attending to future tokens)
        tgt_mask = self.generate_square_subsequent_mask(seq_len)

        # Gets transformer outputs, predicts last token in the sequence
        transformer_out = self.transformer(embeds, memory=embeds, tgt_mask=tgt_mask)
        logits = self.fc(transformer_out)

        loss = None
        if labels is not None:

            # Reshape tensors
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)

            # Shift labels left by 1 and mark last positions as ignored
            labels = torch.roll(labels, shifts=-1)
            labels[::seq_len] = -100

            # This is the loss function!!!!
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

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

    def estimate_parameters(self):
        # Gets estimated parameter count of the model
        d_model = self.config.embedding_dim
        nlayers = self.config.num_layers
        vocab_size = self.config.vocab_size
        ex1 = 12 * d_model**2 * nlayers + 8 * d_model * self.config.hidden_dim * nlayers
        ex2 = (vocab_size * d_model) + (16 * d_model**2) * nlayers

        print(f'This model has {ex1/1e6:.2f}M parameters')
        print(f'This model has {ex2/1e6:.2f}M parameters')
        return None


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',     # Output directory
    num_train_epochs=100,       # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device // usually (256)
    logging_dir='./logs',       # Directory for logs
    logging_steps=100000,
    save_steps=1000,
    save_total_limit=3,
    use_cpu=False,
    fp16=True
)


def main():

    # Loads up model and config
    config = TransformerLMConfig()
    model = TransformerDecoderLM(config)

    load_model = False
    if load_model:
        model.load_state_dict(torch.load('moonshot_alt.pt'))

    stored_train_index = 0
    try:
        with open("number.txt", "r") as file:
            stored_train_index = int(file.read())
    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # This automatically batches + records batch index
    for i, ex in enumerate(dl):
        print(f'{i}th Loop')

        # This prevents retraining on the same batch index. Scales poorly I fear...
        if i != stored_train_index:
            continue

        batch_data = Dataset.from_dict(ex)
        tokenized_datasets = batch_data.map(tokenize_function, batched=True)

        # shuffle the phone number vs. take a random number
        train_dataset = Dataset.from_dict(
            {'input_ids': [x['input_ids'] for x in tokenized_datasets],
             'labels': [x['input_ids'] for x in tokenized_datasets]})

        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Trains, saves, and checkpoints.
        trainer.train()
        torch.save(model.state_dict(), 'moonshot_alt.pt')
        model.load_state_dict(torch.load('moonshot_alt.pt'))
        print(f'Model training loop iteration complete')

        # Writes number to file for completed batch
        with open("number.txt", "w") as file:
            file.write(str(i))
            file.close()


if __name__ == "__main__":
    #main()
    pass

