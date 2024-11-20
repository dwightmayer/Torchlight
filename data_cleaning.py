from datasets import Dataset, load_dataset, concatenate_datasets
from pprint import pprint

ds0 = load_dataset("openbmb/UltraInteract_sft", split='train', streaming=True, trust_remote_code=True) # 151 MB of  code (finetune)
print(ds0)

for i, ex in enumerate(ds0):
    pprint(ex)
    if i == 20:
        break
