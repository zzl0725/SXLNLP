# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        # print(self.index_to_sign)
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = BertTokenizer.from_pretrained(r'E:\AI\课程资料\bert-base-chinese')
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        # print(len(self.tokenizer.encode(sentence)))
        input_id.append(self.tokenizer.encode(sentence))
        with torch.no_grad():
            res = self.model(torch.LongTensor(input_id))[0]
            print(res)
            results = self.decode(res, sentence)
        return results

    def decode(self, res, sentence):
        labels = "".join([str(x) for x in res[1:len(sentence)+1]])
        # print(labels)
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_15.pth")

    sentence = "北京市市长助理、市公安局局长张良基向各位领导同志汇报了北京市公安局圆满完成“两会”安全保卫工作的情况"
    res = sl.predict(sentence)
    print(res)

    sentence = "'从敦煌、当金山、苏干湖、冷湖、花土沟、芒崖、格尔木到昆仑山,行程近2000公里,大漠戈壁、高原鸟岛、雅丹地貌、藏区草原、万丈盐湖、昆仑雪山,奔跑的野驴、迷途的黄羊,还有独特的宗教、民俗、风情……用使人意乱情迷来形容实不为过。"
    res = sl.predict(sentence)
    print(res)
