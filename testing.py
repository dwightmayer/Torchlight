# October 7th, 2024

# Let's fine tune LLAMA2, or die trying...

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchtune.modules.peft import LoRALinear
from bitsandbytes import BitsAndBytesConfig


import torch
import time

start = time.time()

# This is just a random model. Doesn't really matter...
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using 4-bit quantization with bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,   # This enables 4-bit quantization
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # Optional, enables double quantization
        bnb_4bit_quant_type="nf4",        # Quantization type (NF4 is recommended for QLoRA)
        bnb_4bit_compute_dtype=torch.float16  # Reduce computation precision for efficiency
    )
)


# Prepare model for k-bit fine-tuning
model = prepare_model_for_kbit_training(model)

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,                  # LoRA rank
    lora_alpha=32,        # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific attention layers
    lora_dropout=0.1,     # Optional, to prevent overfitting
    bias="none",          # No additional biases are learned
    task_type="CAUSAL_LM" # Language modeling task
)

# Wrap the model with QLoRA
lora_model = get_peft_model(model, lora_config)

# Loads, tokenizes dataset for fine-tuning (subset of wikipedia)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    fp16=True,  # Mixed precision for faster training,
    optim='paged_adamw_8bit' # This reduces space (trust me)
)

# Initialize Trainer with the LoRA model
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
trainer.train()


# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

# Load for inference
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")


end = time.time()
print(f'{end-start:.2f} seconds taken')
