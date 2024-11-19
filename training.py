#!/usr/bin/env python

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from architecture import TransformerDecoderLM, TransformerLMConfig
from model_utils import split_model, reassemble_model


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
dl = DataLoader(dataset, batch_size=1000, shuffle=None, batch_sampler=None, sampler=None)
print('Dataloader created')


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    # maybe increase max length to 512
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',     # Output directory
    num_train_epochs=10,       # Total number of training epochs
    per_device_train_batch_size=256,  # Batch size per device // usually (16)
    logging_dir='./logs',       # Directory for logs
    logging_steps=1000,
    save_steps=1000,
    save_total_limit=3,
    use_cpu=False,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,  # Gradient accumulation
)


def main():

    # Loads up model and config
    config = TransformerLMConfig(tokenizer=tokenizer)
    model = TransformerDecoderLM(config)

    load_model = False
    if load_model:
        model = reassemble_model(model)

    # This txt stores the last batch
    stored_train_index = 0
    pick_up = False
    try:
        with open("number.txt", "r") as file:
            # If desired to start from last saved batch index
            if pick_up:
                stored_train_index = int(file.read())
    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # This automatically batches + records batch index
    for i, ex in enumerate(dl):
        print(f'{i}th Batch')

        # I feel like this doesn't work...
        # This prevents retraining on the same batch index. Scales poorly I fear...
        if i != (stored_train_index): #necessary change to prevent retraining
            continue

        batch_data = Dataset.from_dict(ex)
        tokenized_datasets = batch_data.map(tokenize_function, batched=True)
        # tokenized datasets is a pytorch arrow dataset, whatever that means...

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

        save_model = True
        if save_model:

            split_model(model, num_chunks=10)
            blank_config = TransformerLMConfig(tokenizer=tokenizer)
            blank_model = TransformerDecoderLM(blank_config)
            model = reassemble_model(blank_model)

            # Do i need to do reassemble model?
            #torch.save(model.state_dict(), 'moonshot_alt.pt')
        #model.load_state_dict(torch.load('moonshot_alt.pt'))
        print('vocab size', model.config.vocab_size)
        print(f'Model training loop iteration complete')
        stored_train_index += 1

        # Writes number to file for completed batch
        with open("number.txt", "w") as file:
            file.write(str(i))
            file.close()


if __name__ == "__main__":

    main()

