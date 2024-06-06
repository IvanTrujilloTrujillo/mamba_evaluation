import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Crear la variable de entorno "HF_TOKEN" con el token de Hugging Face

dataset = load_dataset("Trelis/tiny-shakespeare")

dataset.shape

# %%
dataset["train"]

# %%
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# %%
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")

# %%
encodings = tokenizer(dataset["test"]["Text"][:2], return_tensors= "pt", padding=True)

# %%
model.config.get_config_dict("meta-llama/Meta-Llama-3-8B-Instruct")

# %%
def calculate_perplexity(max_length, stride, seq_len):
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls

# %%
# max_length = model.config.n_positions
max_length = model.config.d_model
seq_len = encodings.input_ids.size(1)

# %%
stride = 512
nlls = calculate_perplexity(max_length, stride, seq_len)

ppl = torch.exp(torch.stack(nlls).mean())
print(f"Perplexity: {ppl.item()}")

