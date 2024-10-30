import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from tqdm import tqdm
from data import Data, ValDataBlock
from config import config

# TODO
# Config batch_size num_workers num_device


def train():
    device = torch.device("cuda:0")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(
        device
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(params=model.parameters(), lr=4e-5)
    loss_fn = CrossEntropyLoss()

    dir = ["./datasets/result_pmc_sen_100.txt", "./datasets/result_pmc_pass_100.txt"]
    dataset = Data(*dir)
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader_sen = DataLoader(
        ValDataBlock(*dir, type="sen"),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    test_loader_pass = DataLoader(
        ValDataBlock(*dir, type="pass"),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    with open("./train_log.txt", "w") as f:
        pass
    for s in range(1, 11):  # step 10
        acc_train = []
        acc_val_sen = []
        acc_val_pass = []
        loss_train = []
        loss_val_sen = []
        loss_val_pass = []
        
        with tqdm(data_loader) as loader:
            for i, data in enumerate(loader):
                
                input_text = tokenizer(
                    data[0],
                    data[1],
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                ).to(device)
                label = data[2].to(device)
                out_put = model(**input_text, labels=label)
                loss = out_put["loss"]
                logits = out_put["logits"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # Validation
                with torch.no_grad():
                    acc = torch.sum(torch.argmax(logits, dim=-1) == label) / label.size()[0]
                    if i % 10 == 0:
                        loss_train.append(loss.item())
                        acc_train.append(acc.item())
                        with open("./train_log.txt", "a") as f:
                            f.write(f"Epoch [{s}/{10}] Train ")
                            f.write("Loss %.3f Acc %.3f" % (loss.item(), acc.item()))
                            f.write("\n")
                        loader.set_description(f'Epoch [{s}/{10}] Train')
                        loader.set_postfix(loss = "%.2f" % (loss.item()), acc = "%.2f" % (acc.item()))
                        # break
                        
        with torch.no_grad():
            with tqdm(test_loader_sen) as loader:
                for i, data in enumerate(loader):
                    input_text = tokenizer(
                        data[0],
                        data[1],
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                    ).to(device)
                    label = data[2].to(device)
                    out_put = model(**input_text, labels=label)
                    loss = out_put["loss"]
                    logits = out_put["logits"]
                    loss = torch.sum(loss)
                    acc = torch.sum(torch.argmax(logits, dim=-1) == label) / label.size()[0]
                    if i % 10 == 0:
                        loss_val_sen.append(loss.item())
                        acc_val_sen.append(acc.item())
                        with open("./train_log.txt", "a") as f:
                            f.write(f"Epoch [{s}/{10}] Val_S ")
                            f.write("Loss %.3f Acc %.3f" % (loss.item(), acc.item()))
                            f.write("\n")
                        loader.set_description(f'Epoch [{s}/{10}] Eva_S')
                        loader.set_postfix(loss = "%.2f" % (loss.item()), acc = "%.2f" % (acc.item()))
                        # break
        
        with tqdm(test_loader_pass) as loader:
            for i, data in enumerate(loader):
                input_text = tokenizer(
                    data[0],
                    data[1],
                    return_tensors="pt",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                ).to(device)
                label = data[2].to(device)
                out_put = model(**input_text, labels=label)
                loss = out_put["loss"]
                logits = out_put["logits"]
                loss = torch.sum(loss)
                acc = torch.sum(torch.argmax(logits, dim=-1) == label) / label.size()[0]
                if i % 10 == 0:
                    loss_val_pass.append(loss.item())
                    acc_val_pass.append(acc.item())
                    with open("./train_log.txt", "a") as f:
                        f.write(f"Epoch [{s}/{10}] Val_S ")
                        f.write("Loss %.3f Acc %.3f" % (loss.item(), acc.item()))
                        f.write("\n")
                    loader.set_description(f'Epoch [{s}/{10}] Eva_P')
                    loader.set_postfix(loss = "%.2f" % (loss.item()), acc = "%.2f" % (acc.item()))
                    # break

        model.save_pretrained(f"./models/{s}")
        with open(f"./models/{s}/loss_train.txt", "w") as f:
            f.write(str(loss_train))
        with open(f"./models/{s}/acc_train.txt", "w") as f:
            f.write(str(acc_train))
        with open(f"./models/{s}/loss_val_sen", "w") as f:
            f.write(str(loss_val_sen))
        with open(f"./models/{s}/acc_val_sen", "w") as f:
            f.write(str(acc_val_sen))
        with open(f"./models/{s}/loss_val_pass", "w") as f:
            f.write(str(loss_val_pass))
        with open(f"./models/{s}/acc_val_pass", "w") as f:
            f.write(str(acc_val_pass))


if __name__ == "__main__":
    train()
