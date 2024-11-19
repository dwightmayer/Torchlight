import torch
import os
# Utils to split, reassemble model

def split_model(model_path, num_chunks):
    # Splits model for storage
    state_dict = model_path.state_dict()
    state_dict_items = list(state_dict.items())
    #state_dict_items = list(model_path.items())
    chunk_size = len(state_dict_items) // num_chunks

    for i in range(num_chunks):
        chunk = dict(state_dict_items[i * chunk_size:(i + 1) * chunk_size])
        chunk_path = f"model_chunks/moonshot_chunk_{i + 1}.pt"
        torch.save(chunk, chunk_path)

def reassemble_model(empty_model):
    # Reassembles model for prediction
    reconstructed_state_dict = {}
    chunk_files = sorted(
        [f for f in os.listdir('model_chunks') if f.startswith('moonshot_chunk_') and f.endswith('.pt')])
    num_chunks = len(chunk_files)

    for i in range(1, num_chunks+1):
        chunk_path = f'model_chunks/moonshot_chunk_{i}.pt'
        chunk = torch.load(chunk_path)
        reconstructed_state_dict.update(chunk)

    model = empty_model
    missing_keys, unexpected_keys = model.load_state_dict(reconstructed_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys during loading: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys during loading: {unexpected_keys}")

    print("Model successfully reassembled!")
    return model



