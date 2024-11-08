# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path

        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)

        # self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # self.create_dataset()
        self.load()

    # def create_dataset(self):
        # data = pd.read_csv(r"E:\AI\课程资料\复习\07\homework\文本分类练习数据集\文本分类练习.csv")
        # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        # # 划分为训练集和测试集（80%训练，20%测试）
        # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        # # 存储为 JSON 文件，确保汉字正常显示
        # train_data.to_json("train_data.json", orient='records', lines=True, force_ascii=False)
        # test_data.to_json("test_data.json", orient='records', lines=True, force_ascii=False)
        # with open("train_data.json", "r", encoding="utf8") as f:
        #     total_title = []
        #     for line in f:
        #         line = json.loads(line)
        #         title = line["review"]
        #         total_title.append(title)
        #     total_characters = sum(len(text) for text in total_title)
        #     avg_length = total_characters // len(total_title)
        #     self.config["avg_length"] = avg_length


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if line.startswith("0,"):
                    label = 0
                elif line.startswith("1,"):
                    label = 1
                else:
                    continue
                title = line[2:].strip()
                # line = json.loads(line)
                # label = line["label"]
                # title = line["review"]
                # print(label, title)
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], padding='max_length', truncation=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        # input_id = input_id[:self.config["avg_length"]]
        # input_id += [0] * (self.config["avg_length"] - len(input_id))
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("train_data.txt", Config)
    # # 迭代 DataLoader，查看数据内容
    for batch in dg:
        print(batch)
