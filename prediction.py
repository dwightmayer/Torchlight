import torch

from transformers import AutoTokenizer

# Load a pretrained tokenizer or the one you used for training
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

from testing import TransformerDecoderLM, TransformerLMConfig

# Instantiate the model and load the weights
config = TransformerLMConfig()
model = TransformerDecoderLM(config)
model.load_state_dict(torch.load('moonshot_alt.pt'))
model.eval()  # Set to evaluation mode

def predict_next_word(input_text, model, tokenizer):
    # Tokenize input and convert to tensor
    inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=5, return_tensors='pt')

    # Dictionary fields
    inputs.pop('attention_mask', None)
    inputs.pop('token_type_ids', None)

    # Run the model and get logits
    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs)
    print(type(outputs))

    # Get the index of the predicted next token
    next_token_logits = outputs['logits'][:, -1, :]
    predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

    # Decode to get the word
    predicted_word = tokenizer.decode(predicted_token_id)

    return predicted_word


# Example usage
input_text = "The weather today is"
predicted_word = predict_next_word(input_text, model, tokenizer)
print(f"Predicted next word: {predicted_word}")


