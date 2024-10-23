# October 7th, 2024

# Let's fine tune LLAMA2, or die trying...

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
import regex as re
import time

## Get data

# 235MB subset of wikipedia, allegedly preprocessed try 2-5 epochs on whole set
dataset = load_dataset("wikipedia", "20220301.simple")


# Sample data
text = "Four score and seven years ago..."

with open('training/moby-dick.txt', 'r') as f:
    text = f.read()


def preprocess_text(txt):
    """Clean and tokenize text. Nothing super special."""
    # Convert to lowercase, remove whitespace nad special characters (excluding punctuation)
    txt = txt.lower()
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'[^a-z0-9\s.,!?-]', '', txt)

    # Split into tokens
    text_tokens = txt.split()
    return text_tokens


# Tokenization and vocabulary creation
tokens = preprocess_text(text)
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
    def __init__(self,
                 vocab_size=None,
                 embedding_dim=64,
                 hidden_dim=128,
                 n_heads=8,
                 num_layers=4,
                 sequence_length=5,
                 **kwargs):

        super().__init__(**kwargs)
        # Set defaults if not provided

        # // Set to <100 and then run through small sample text to ensure system catches errors
        self.vocab_size = vocab_size if vocab_size is not None else 30000
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
        logits = self.fc(transformer_out[:, -1, :])

        loss = None
        if labels is not None:
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


# Instantiate the model and configuration. This restates the default arguments of the class. Problematic?
old_config = TransformerLMConfig(vocab_size=vocab_size,
                             embedding_dim=64,
                             hidden_dim=128,
                             n_heads=8,
                             num_layers=2)

config = TransformerLMConfig()
model = TransformerDecoderLM(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',     # Output directory
    num_train_epochs=100,       # Total number of training epochs
    per_device_train_batch_size=256,  # Batch size per device
    logging_dir='./logs',       # Directory for logs
    logging_steps=10,
    save_steps=1000,
    save_total_limit=3,
    use_cpu=True # CHANGE THIS WHEN CUDA IS AVAILABLE
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()
print(f'VOCAB SIZE: {(vocab_size)}')

