import os
import json
import pandas as pd
"""
examples:
https://github.com/imoneoi/openchat#installation
{"items":[{"role":"user","content":"Hello","weight":0.0},{"role":"assistant","content":"Hi","weight":1.0},{"role":"user","content":"How are you today?","weight":0.0},{"role":"assistant","content":"I'm fine.","weight":1.0}],"system":""}
{"items":[{"role":"user","content":"Who are you?","weight":0.0},{"role":"assistant","content":"I'm OpenChat.","weight":1.0}],"system":"You are a helpful assistant named OpenChat."}
"""

def items_create(user_phrases: list[str], assistant_phrases: list[str]):
    phrases_seq = []
    for up, ap in zip(user_phrases, assistant_phrases):
        phrases_seq.append({"role":"user","content": up,"weight":0.0})
        phrases_seq.append({"role":"assistant","content":ap,"weight":1.0})
    return {"items": phrases_seq,"system":""}


queries_df = pd.read_csv(os.path.join(os.getcwd(), "data", "queries_with_short_queries_clearing.csv"), sep="\t")
queries_dicts = queries_df.to_dict(orient="records")


train_dict = items_create(queries_df["Query"], queries_df["ShortQuery"])

with open(os.path.join(os.getcwd(), "data", "train_dataset.json"), "w") as f:
    json.dump(train_dict, f, ensure_ascii=False)