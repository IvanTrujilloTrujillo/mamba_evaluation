# %%
import torch
from transformers import AutoTokenizer, MambaForCausalLM, AutoModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import os

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
data = pd.read_csv('data/disaster_tweets/train.csv')

# %%
data.head()

# %%
data = data.sample(500)

# %%
data.shape

# %%
data[data["target"] == 1].shape[0], data[data["target"] == 0].shape[0]

# %%
strategy = "zero_shot"

# %%
model = MambaLMHeadModel.from_pretrained(f"havenhq/mamba-chat").to(device)
# model = AutoModel.from_pretrained(f"havenhq/mamba-chat", ignore_mismatched_sizes=True).to(device)
# model = MambaLMHeadModel.from_pretrained(os.path.expanduser("state-spaces/mamba-{model_size}"), device="cuda", dtype=torch.bfloat16)

# %%
tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat")

# %%
prompt_template_zero_shot = """
Instructions:

You have to analyze the following tweet and to determine if it speaks about a real desaster or not. Answer with "1" if the tweet speaks about a real disaster and with "0" if not. Don't add any other information in your answer.

--------------------------
Tweet:

{text}
--------------------------
Your answer (only a "1" or a "0"):
"""

# %%
prompt_template_few_shot = """
Instructions:
Your task is to analyze the following tweet and determine if it is talking about a real disaster. A real disaster can include, but is not limited to, events such as earthquakes, hurricanes, fires, floods, major accidents, etc. If the tweet refers to a real disaster, respond with 1. If not, respond with 0.

Your response should only be the number 1 or 0.

Considerations:
Real Disasters: Significant events that impact people, property, or the environment.
Not Disasters: Personal opinions, jokes, fake news, or events that do not qualify as a disaster.

Examples:
Tweet: "A 7.5 magnitude earthquake has shaken the city, causing significant damage and injuries."
Expected Response: 1

Tweet: "I'm so tired that my house looks like a disaster after last night's party!"
Expected Response: 0

Tweet: "Uncontrolled wildfire in the north of the country. Evacuate immediately."
Expected Response: 1

Tweet: "It rained a lot yesterday, but today is sunny and beautiful."
Expected Response: 0

Tweet to Analyze:
Tweet: "{text}"

Response:
Result (1 or 0):
"""

# %%
prompt_template = prompt_template_zero_shot if strategy == "zero_shot" else prompt_template_few_shot

# %%
predictions = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['text'])
    messages = [dict(role="user", content=prompt)]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    out = model.generate(input_ids=input_ids, max_length=2000, temperature=0.9, top_p=0.7, eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(out)
    try:
        predictions.append(int(decoded))
    except:
        print(f"{index}: {decoded}")
    if index % 50:
        print(index)

# %%
predictions = []
with torch.no_grad():
    for index, row in data.iterrows():
        prompt = prompt_template.format(text=row['text'])
        encodings = tokenizer(prompt, return_tensors="pt")
        input_ids = encodings.to(device)
        #outputs = model(**input_ids, max_new_tokens=1)
        outputs = model(**input_ids)
        #p = tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
        p = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
        # predictions.append(p)
        try:
            predictions.append(int(p))
        except:
            print(f"{index}: {p}")
        if index % 50:
            print(index)

# %%
data["predictions"] = predictions

# %%
f1_score(data["target"], data["predictions"])


