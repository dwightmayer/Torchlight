import torch
import torch.nn as nn

from transformers import AutoTokenizer
from testing import TransformerDecoderLM, TransformerLMConfig

# Load a pretrained tokenizer or the one you used for training
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Instantiate the model and load the weights from previous scripts
config = TransformerLMConfig()
model = TransformerDecoderLM(config)
model.load_state_dict(torch.load('moonshot_alt.pt'))
model.eval()


def generate_next_tokens(
        model: nn.Module,
        text: str,
        tokenizer,
        max_new_tokens: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> str:

    model.eval()

    tokens = tokenizer(text,
                       return_tensors='pt',
                       padding=True,
                       truncation=True,
                       max_length=model.config.sequence_length)

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Gets outputs, and logits of last output
            outputs = model(input_ids, attention_mask)['logits']
            last_token_logits = outputs[-1]

            # normalizes inputs. // Can call argmax directly if you want
            next_token_probs = torch.softmax(last_token_logits, dim=-1) # changeed from 0


            next_token = torch.argmin(next_token_probs)# trying on argmin. think [SEP] dominates?
            # trying for a more varied sample.
            #next_token = torch.multinomial(next_token_probs, num_samples=1)

            generated.append(next_token.item())

            # Gets input ids and updates attention mask
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

            # Truncates above max length
            if input_ids.size(1) >= model.config.sequence_length:
                input_ids = input_ids[:, -model.config.sequence_length:]
                attention_mask = attention_mask[:, -model.config.sequence_length:]

    # when skip=True, I get '' as my pred, skip=False gets me [SEP] x5 // unsure why.
    generated_tokens = tokenizer.decode(generated, skip_special_tokens=False)
    print(generated_tokens)
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
    print("Logit for [SEP]:", last_token_logits[sep_token_id].item())

    return generated_tokens


def main():
    # Example usage
    input_text = "The weather today is"
    predicted_word = generate_next_tokens(model=model, text=input_text, tokenizer=tokenizer)
    print(f"Predicted next word: THE WEATHER TODAY IS: //  {predicted_word}")


if __name__ == "__main__":
    main()

