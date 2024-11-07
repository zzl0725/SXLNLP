#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
"""
基于pytorch的LSTM语言模型
"""
class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)
#加载语料
def load_corpus(path):
    input_data = []
    label_data = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            input_data.append(title)
            label_data.append(content)
    return input_data, label_data

def build_dataset(input_data, label_data, tokenizer, max_length=50):
    dataset_x = []
    dataset_y = []
    masks = []
    for input_text, label in zip(input_data, label_data):
        # 编码输入和标签
        input_id = 'CLS' + input_text + '[SEP]' + label
        input_id = tokenizer.encode(input_id, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)
        label_id = label + '[SEP]'
        label_id = tokenizer.encode(label_id, add_special_tokens=False, padding='max_length', truncation=True, max_length=max_length)
        # 创建填充向量，这里要确保它的长度为 max_length
        padding_vector = torch.full((len(input_text),), -100)
        # 拼接填充向量与标签 ID
        result_label = torch.cat((padding_vector, torch.tensor(label_id)))
        # 确保结果长度为 max_length
        result_label = result_label[:max_length]  # 截取到最大长度
        if result_label.size(0) < max_length:
            # 如果结果长度不足，则填充到 max_length
            result_label = torch.cat((result_label, torch.full((max_length - result_label.size(0),), -100)))
        dataset_x.append(torch.tensor(input_id))  # 转为张量
        dataset_y.append(result_label)  # result_label 已经是张量
        masks.append(create_mask(len(input_text), len(label_id), max_length))  # 传入长度
    # 将数据堆叠为张量，确保每个张量的形状相同
    return torch.stack(dataset_x), torch.stack(dataset_y), torch.stack(masks)


def create_mask(input_len, label_len, max_length=50):
    total_len = min(input_len + label_len, max_length)
    mask = torch.zeros((max_length, max_length), dtype=torch.int)

    # 对于输入部分
    mask[:input_len+2, :input_len+2] = 1
    mask[input_len+2:total_len, :input_len+2] = 1

    # 对于标签部分
    if label_len > 1:
        mask[input_len+2:total_len, input_len+2:total_len] = torch.tril(
            torch.ones((label_len-1, label_len-1), dtype=torch.int)
        )[:max_length - input_len-2, :max_length - input_len-2]  # 确保不会超出 max_length

    return mask

#建立模型
def build_model(pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings, add_special_tokens=False)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def train(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 8      #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    learning_rate = 0.0001  #学习率
    pretrain_model_path = r'E:\AI\课程资料\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    input_data, label_data = load_corpus('sample_data.json')     #加载语料
    model = build_model(pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(13):
            x, y, mask = build_dataset(input_data, label_data, tokenizer) #构建一组训练样本
            x = x[i * batch_size: (i + 1) * batch_size]
            y = y[i * batch_size: (i + 1) * batch_size]
            mask = mask[i * batch_size: (i + 1) * batch_size]  # 构建一组 mask
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)
