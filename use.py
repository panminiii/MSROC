from transformers import BertForSequenceClassification, BertTokenizer
import torch

class Schedule():
    def __init__(self, model_dir, device):
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = device
        
    def tokenize(self, query, text):
        return self.tokenizer(
                    query,
                    text,
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                ).to(self.device)
    
    def score(self, query, text):
        input_s = self.tokenize(query, text)
        # print(input_s)
        # print(type(input_s["input_ids"]))
        out_put = self.model(**input_s)
        logits = out_put["logits"]
        label = torch.argmax(logits, dim=-1)
        
        conf = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        return conf


if __name__ == "__main__":
    
    # Data =========================================================
    with open("./datasets/result_pmc_sen_100.txt") as f:
        data_sen = f.readlines()[:20]
    data_query = []
    data_doc = []
    for i in data_sen:
        d = i.split("\t")
        if len(d) == 2:
            data_query.append(d[0])
            data_doc.append(d[1])
    
    # Use Multy data
    schedule = Schedule("./models/6/", torch.device("cuda:0"))
    out = schedule.score(data_query, data_doc)
    print("Label", out[1])
    print("Confidence", out[0])

    # Data =========================================================
    with open("./datasets/result_pmc_sen_100.txt") as f:
        data_sen = f.readlines()[:1]
    data_query = []
    data_doc = []
    for i in data_sen:
        d = i.split("\t")
        if len(d) == 2:
            data_query.append(d[0])
            data_doc.append(d[1])
    # Use Sing data
    out = schedule.score(data_query, data_doc)
    print("Label", out[1])
    print("Confidence", out[0])
    