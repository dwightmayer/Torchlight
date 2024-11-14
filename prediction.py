import torch
import torch.nn as nn
from transformers import AutoTokenizer
from training import TransformerDecoderLM, TransformerLMConfig

# Load a pretrained tokenizer or the one you used for training
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Instantiate the model and load the weights from previous scripts
config = TransformerLMConfig()
model = TransformerDecoderLM(config)
device = torch.device("cpu")
model.load_state_dict(torch.load('moonshot_alt.pt', map_location=device))
model.eval()


def generate_next_tokens(
        model: nn.Module,
        text: str,
        tokenizer,
        max_new_tokens: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> str:

    model.eval()

    tokens = tokenizer(text,
                       return_tensors='pt',
                       padding='max_length', # changed from TRUE
                       truncation=True,
                       max_length=model.config.sequence_length)

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Gets outputs, and logits of last output
            outputs = model(input_ids, attention_mask)['logits']
            last_token_logits = outputs[-1, :]

            # normalizes inputs. // Can call argmax directly if you want
            next_token_probs = torch.softmax(last_token_logits, dim=-1) # changed from 0
            next_token = torch.argmax(next_token_probs)

            # trying for a more varied sample.
            generated.append(next_token.item())

            # Gets input ids and updates attention mask
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

            # Truncates above max length
            if input_ids.size(1) >= model.config.sequence_length:
                input_ids = input_ids[:, -model.config.sequence_length:]
                attention_mask = attention_mask[:, -model.config.sequence_length:]

            k = 10
            top_probs, top_indices = torch.topk(last_token_logits, k)

            # Extract the top logit and corresponding tokens
            first_largest_logit = top_probs[0].item()
            second_largest_logit = top_probs[1].item() if k > 1 else None
            third_largest_logit = top_probs[2].item() if k > 2 else None

            # Print the largest and second largest logit with their tokens
            first_token_string = tokenizer.convert_ids_to_tokens(top_indices[0].item())
            second_token_string = tokenizer.convert_ids_to_tokens(
                top_indices[1].item()) if second_largest_logit is not None else "N/A"
            third_logit_string = tokenizer.convert_ids_to_tokens(
                top_indices[2].item()) if third_largest_logit is not None else "N/A"

            print(round(first_largest_logit, 2), first_token_string)
            print(round(second_largest_logit, 2), second_token_string)
            print(round(third_largest_logit, 2), third_logit_string)

    # when skip=True, I get '' as my prediction, skip=False gets me [SEP] x5 // unsure why.
    generated_tokens = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_tokens


def main():
    # Example usage
    input_text = "the quick brown fox jumped over the lazy "
    predicted_word = generate_next_tokens(model=model, text=input_text, tokenizer=tokenizer)
    # print(f"Predicted next word: //  {predicted_word}")
    # getting lots of PAD characters
    model.estimate_parameters()


if __name__ == "__main__":
    main()


