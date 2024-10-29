import torch
import torch.nn as nn

from transformers import AutoTokenizer
from training import TransformerDecoderLM, TransformerLMConfig

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets

import evaluate


# Load a pretrained tokenizer or the one you used for training
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Instantiate the model and load the weights from previous scripts
config = TransformerLMConfig()
model = TransformerDecoderLM(config)
model.load_state_dict(torch.load('moonshot_alt.pt'))
model.eval()

with open('training/sherlock-holmes.txt', 'r') as f:
    text = f.read()

sentences = text.split('.')
validation_texts = sentences

validation_inputs = tokenizer(validation_texts, return_tensors="pt", padding=True, truncation=True)
# Create a DataLoader for validation
validation_loader = DataLoader(validation_inputs['input_ids'], batch_size=2)

# Initialize variable to compute total loss
total_loss = 0.0
num_batches = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    for batch in validation_loader:

        print(batch.shape)

        # Move the batch to the appropriate device (CPU or GPU)
        batch = batch.to(model.device)

        # Forward pass to get the loss
        outputs = model(batch, labels=batch)  # Use labels for loss calculation
        loss = outputs.loss

        # Accumulate total loss
        total_loss += loss.item()
        num_batches += 1

        # Calculate perplexity for the current batch
        perplexity = torch.exp(loss)

        # Print the loss and perplexity for the current batch
        print(f"Batch {num_batches}: Loss = {loss.item():.4f}, Perplexity = {perplexity.item():.4f}")

# Calculate average loss and perplexity over all batches
average_loss = total_loss / num_batches
average_perplexity = torch.exp(torch.tensor(average_loss))

print(f"\nAverage Loss: {average_loss:.4f}")
print(f"Average Perplexity: {average_perplexity.item():.4f}")

ds = load_dataset("bookcorpus", trust_remote_code=True)
validation_set = ds['train'][0:100]

evaluation_dataset = None
metric = None
for model_inputs, gold_standards in evaluation_dataset:
    predictions = model(model_inputs)
    metric.add_batch(references=gold_standards, predictions=predictions)
metric.compute()

