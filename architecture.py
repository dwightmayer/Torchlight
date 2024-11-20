# This script will be the architecture and class definition of the model.
# Training will be done elsewhere from now on to avoid clutter
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np

# get a model.yaml file

# Model Config class for Hugging Face compatibility
class TransformerLMConfig(PretrainedConfig):
    def __init__(self,
                 tokenizer,  # I might make tokenizer an attribute
                 embedding_dim=256,  # setting this from 128 to 192... does it fit?
                 hidden_dim=512,
                 n_heads=16,
                 num_layers=16,
                 sequence_length=32,
                 **kwargs):

        super().__init__(**kwargs)
        # Set defaults if not provided

        # Sets vocab size
        #self.tokenizer = tokenizer
        vocab_size = tokenizer.vocab_size
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length


        self.hidden_dropout_prob = 0.2,
        self.attention_probs_dropout_prob = 0.2,

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

        # Final linear layer for word prediction.py and checkpointing
        self.fc = nn.Linear(config.embedding_dim, config.vocab_size)
        self.supports_gradient_checkpointing = True

    def forward(self, input_ids, labels=None, use_cache=None):

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Creates embeddings with positional encodings :)
        embeds = self.embedding(input_ids) + self.pos_encoding[:seq_len, :]

        # Create causal mask (upper triangular to prevent checking out future tokens)
        tgt_mask = self.generate_square_subsequent_mask(seq_len)

        # Disable cache if gradient checkpointing
        if use_cache is None:
            use_cache = not self.training or not self.supports_gradient_checkpointing

        # Uses gradient checkpointing if enabled, normal processing if not:
        if self.supports_gradient_checkpointing and self.training:
            transformer_out = torch.utils.checkpoint.checkpoint(
                    self.transformer, tgt=embeds, memory=embeds, tgt_mask=tgt_mask,
                    use_reentrant=False  # Add this for newer PyTorch versions // idk
                )
        else:
            transformer_out = self.transformer(embeds, memory=embeds, tgt_mask=tgt_mask)

        # Gets transformer outputs, predicts last token in the sequence
        # transformer_out = self.transformer(embeds, memory=embeds, tgt_mask=tgt_mask)
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

    # Cheeky new set of methods to ensure compatibility HF checkpoints
    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing (for transformer layers!!!)"""
        if isinstance(module, nn.TransformerDecoder):
            module.use_checkpoint = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        # Error handling if GC / model class tweaking
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        self.apply(lambda m: self._set_gradient_checkpointing(m, True))

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        # Should I add more error handling here?
        self.apply(lambda m: self._set_gradient_checkpointing(m, False))

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
        # Variables, equations for different parameter counts
        d_model = self.config.embedding_dim
        n_layers = self.config.num_layers
        vocab_size = self.config.vocab_size
        ex1 = 12 * d_model**2 * n_layers + 8 * d_model * self.config.hidden_dim * n_layers
        ex2 = (vocab_size * d_model) + (16 * d_model**2) * n_layers

        print(f'This model has {(min(ex1, ex2))/1e6:.1f}M - {max((ex1, ex2))/1e6:.1f}M parameters')

