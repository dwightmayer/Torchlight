import torch
import torch.nn as nn
from transformers import AutoTokenizer
from training import TransformerDecoderLM, TransformerLMConfig

# Load a pretrained tokenizer
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
            # This needs last_token_logits to be above 10, usually should be the case...
            top_probs, top_indices = torch.topk(last_token_logits, k)
            for i in range(k):
                ith_largest_logit = top_probs[i].item() if i <= len(top_probs) else None
                ith_token_string = tokenizer.convert_ids_to_tokens(top_indices[i].item()
                                                                   if ith_largest_logit is not None else "N/A")

                print(f'{i+1}th most likely token prediction')
                print(ith_token_string,  round(ith_largest_logit, 2))
                print()

    # Try out changing this. Manifesting that I get good results from it.
    generated_tokens = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_tokens


def main():
    # Example usage
    input_text = "i like eating "
    predicted_word = generate_next_tokens(model=model, text=input_text, tokenizer=tokenizer)
    # getting lots of PAD characters
    model.estimate_parameters()
    # some of this stuff takes so much time to install that my head explodes instantly


if __name__ == "__main__":
    main()


