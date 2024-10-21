# October 7th, 2024

# Let's fine tune LLAMA2, or die trying...

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
import time

# Sample data
text = "Four score and seven years ago..."

with open('sample_text.txt', 'r') as f:
    text = f.read()


# Tokenization and vocabulary creation
tokens = text.lower().split()
vocab = sorted(set(tokens))
vocab_size = len(vocab)

# Create word to index and index to word mappings
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# Convert text to sequence of indices
input_sequence = [word_to_idx[word] for word in tokens]
sequence_length = 5  # Window size

# Create sequences
def create_sequences(input_sequence, sequence_length):
    sequences = []
    for i in range(len(input_sequence) - sequence_length):
        seq = input_sequence[i:i + sequence_length]
        target = input_sequence[i + sequence_length]
        sequences.append({'input_ids': seq, 'labels': target})
    return sequences


# Prepare dataset
train_data = create_sequences(input_sequence, sequence_length)
train_dataset = Dataset.from_dict({'input_ids': [x['input_ids'] for x in train_data],
                                   'labels': [x['labels'] for x in train_data]})


# Model Config class for Hugging Face compatibility
class TransformerLMConfig(PretrainedConfig):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_heads, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length


# Define the Transformer Model using Hugging Face's PreTrainedModel
class TransformerLM(PreTrainedModel):
    config_class = TransformerLMConfig

    def __init__(self, config):
        super().__init__(config)

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional encoding
        self.pos_encoding = self.generate_positional_encoding(config.embedding_dim, config.sequence_length)

        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=config.embedding_dim,
                                                       nhead=config.n_heads,
                                                       dim_feedforward=config.hidden_dim)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=config.num_layers)

        # Final linear layer for word prediction
        self.fc = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, input_ids, labels=None):
        embeds = self.embedding(input_ids) + self.pos_encoding[:input_ids.size(1), :]
        transformer_out = self.transformer(embeds)
        logits = self.fc(transformer_out[:, -1, :])

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def generate_positional_encoding(self, dim, max_len):
        pos_enc = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc.unsqueeze(0)


# Instantiate the model and configuration
config = TransformerLMConfig(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, n_heads=8, num_layers=2)
model = TransformerLM(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',     # Output directory
    num_train_epochs=100,       # Total number of training epochs
    per_device_train_batch_size=2,  # Batch size per device
    logging_dir='./logs',       # Directory for logs
    logging_steps=10,
    no_cuda=True,
    save_steps=1000,
    save_total_limit=3

)

# Define Trainer
trainer = Trainer(
    model=model,                 # The model to train
    args=training_args,          # Training arguments
    train_dataset=train_dataset, # Training dataset
)

# Train the model
trainer.train()
