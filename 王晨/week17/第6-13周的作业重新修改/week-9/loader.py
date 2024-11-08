# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(r'E:\AI\课程资料\bert-base-chinese')
        self.load()

    def load(self):
        self.data = []
        with open(self.path, "r", encoding="utf-8") as f:
            segments = f.read().split("\n\n")
            for sentence in segments:
                sentences = []
                labels = [8]
                sentence = sentence.split("\n")
                for word in sentence:
                    if word.strip() == '':
                        continue
                    else:
                        char, label = word.strip().split(" ")
                        labels.append(self.schema[label])
                        sentences.append(char)
                self.sentences.append(''.join(sentences))
                # print(self.sentences)
                # input_ids = self.encode_sentence(sentences)
                input_ids = self.tokenizer.encode(sentences, padding='max_length', truncation=True, max_length=self.config["max_length"],)
                labels = self.padding(labels, -1)
                # print(len(input_ids), len(labels))
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])


    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def encode_sentence(self, sentence, padding=True):
        input_id = []
        for char in sentence:
            if char in self.vocab:
                input_id.append(self.vocab[char])
            else:
                input_id.append(self.vocab['[UNK]'])
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_ids, padding_token=0):
        max_length = self.config["max_length"]
        input_ids = input_ids[:max_length]
        input_ids += [padding_token] * (max_length - len(input_ids))
        return input_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            vocab[line.strip()] = index + 1
    # 将字典写入 JSON 文件
    with open('vocab.json', 'w', encoding='utf-8') as json_file:
        json.dump(vocab, json_file, ensure_ascii=False, indent=2)
    return vocab


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("./ner_data/train", Config)
    # print(dg[0])
