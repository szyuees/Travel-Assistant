#generate lora_train.csv
import json, csv
data = json.load(open("data/faq_data.json"))
with open("data/lora_train.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["question","answer"])
    for d in data: w.writerow([d["question"], d["answer"]])