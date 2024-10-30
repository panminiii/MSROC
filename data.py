import torch
from torch.utils.data import Dataset, DataLoader
from random import randint


def sample_no_repeat_list(min_num, max_num, list_len):
    list_l = []
    if (max_num - min_num + 1) < list_len:
        print("Len too big")
        return []
    while True:
        if len(list_l) >= list_len:
            break
        step_num = randint(min_num, max_num)
        if step_num not in list_l:
            list_l.append(step_num)
    return list_l


def pre_porcess_data(data_sen_dir, data_pass_dir):
    with open(data_sen_dir) as f:
        data_sen = f.readlines()
    with open(data_pass_dir) as f:
        data_pass = f.readlines()
    data_sen_d = {}
    data_pass_d = {}
    last_query = ""
    for i in data_sen:
        i = list(i.split("\t"))
        if len(i) == 2:
            query = i[0]
            if last_query == query:
                data_sen_d[last_query].append(i[1])
            else:
                last_query = query
                data_sen_d[last_query] = []
    last_query = ""
    for i in data_pass:
        i = list(i.split("\t"))
        if len(i) == 2:
            query = i[0]
            if last_query == query:
                data_pass_d[last_query].append(i[1])
            else:
                last_query = query
                data_pass_d[last_query] = []
    return data_sen_d, data_pass_d


class Data(Dataset):
    def __init__(self, data_sen_dir, data_pass_dir):
        super().__init__()
        # train 0: 22
        self.data_sen, self.data_pass = pre_porcess_data(data_sen_dir, data_pass_dir)
        querys = list(self.data_sen.keys())
        self.train_query = querys[0:22]
        self.val_query = querys[22:26]
        self.test_query = querys[26:30]
        with open("./query_spilt.txt", "w") as f:
            f.write("Query split: \n")
        with open("./query_spilt.txt", "a") as f:
            f.write("train " + str(self.train_query) + "\n")
            f.write("val " + str(self.val_query) + "\n")
            f.write("test " + str(self.test_query) + "\n")
        self.index = []
        self.total_len = 0
        for i in self.train_query:
            len_l = len(self.data_sen[i]) + len(self.data_pass[i])
            self.index.append(len_l)
            self.total_len += len_l

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        query, data = self.data_index(index)
        if randint(0, 1):
            # TODO 长短句的数量差距，对实验结果（负例）的影响
            _, data = self.data_index(randint(0, self.total_len - 1))
            return query, data, 0  # NEGETIVE
        else:
            return query, data, 1  # POSITIVE

    def data_index(self, index):
        if index > self.total_len:
            exit("Error" + " {} | {}".format(index, self.total_len))
        len_l = 0
        for i, l in enumerate(self.index):
            len_l += l
            if len_l >= index:
                query = self.train_query[i]
                pice_index = len_l - index - 1
                first_pice = len(self.data_sen[query])
                if first_pice > pice_index:
                    data = self.data_sen[query][pice_index]
                    return query, data
                else:
                    pice_index = pice_index - first_pice
                    data = self.data_pass[query][pice_index]
                    return query, data


class ValDataBlock(Dataset):
    def __init__(self, data_sen_dir, data_pass_dir, type):
        super().__init__()
        # val 22: 26
        if type == "sen":
            self.data, _ = pre_porcess_data(data_sen_dir, data_pass_dir)
        elif type == "pass":
            _, self.data = pre_porcess_data(data_sen_dir, data_pass_dir)
        querys = list(self.data.keys())
        self.val_query = querys[22:26]
        self.index = []
        self.total_len = 0
        for i in self.val_query:
            len_l = len(self.data[i])
            self.index.append(len_l)
            self.total_len += len_l

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        query, data = self.data_index(index)
        if randint(0, 1):
            # TODO 长短句的数量差距，对实验结果（负例）的影响
            _, data = self.data_index(randint(0, self.total_len - 1))
            return query, data, 0  # NEGETIVE
        else:
            return query, data, 1  # POSITIVE

    def data_index(self, index):
        if index > self.total_len:
            exit("Error" + " {} | {}".format(index, self.total_len))
        len_l = 0
        for i, l in enumerate(self.index):
            len_l += l
            if len_l >= index:
                query = self.val_query[i]
                pice_index = len_l - index - 1
                data = self.data[query][pice_index]
                return query, data


if __name__ == "__main__":
    print("/**************** Test Sample *******************/")
    result1 = sample_no_repeat_list(0, 29, 30)
    print(
        "Train: \t",
        result1[0:22],
        "\n",
        "Val: \t",
        result1[22:26],
        "\n",
        "Test: \t",
        result1[26:30],
    )

    print("/**************** Test Preprocess *******************/")
    resul2 = pre_porcess_data(
        "./datasets/result_pmc_sen_100.txt", "./datasets/result_pmc_pass_100.txt"
    )
    print(resul2[0].keys(), resul2[0].keys().__len__())

    print("/**************** Test Loading *******************/")
    dir = ["./datasets/result_pmc_sen_100.txt", "./datasets/result_pmc_pass_100.txt"]
    loader = DataLoader(Data(*dir), batch_size=2, shuffle=True, num_workers=2)
    for i, d in enumerate(loader):
        print(d)
        if i == 9:
            break
        pass

    print("/**************** Test test *******************/")
    dir = ["./datasets/result_pmc_sen_100.txt", "./datasets/result_pmc_pass_100.txt"]
    loader = DataLoader(ValDataBlock(*dir, type="sen"), batch_size=2, num_workers=2)
    for i, d in enumerate(loader):
        print(d)
        if i == 9:
            break
        pass

    print("/**************** Test pass *******************/")
    dir = ["./datasets/result_pmc_sen_100.txt", "./datasets/result_pmc_pass_100.txt"]
    loader = DataLoader(ValDataBlock(*dir, type="pass"), batch_size=2, num_workers=2)
    for i, d in enumerate(loader):
        print(d)
        if i == 9:
            break
        pass