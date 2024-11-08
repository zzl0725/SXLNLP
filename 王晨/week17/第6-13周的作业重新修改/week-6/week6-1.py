import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"E:\AI\课程资料\第六周 预训练模型\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数

token_embeddings = vocab * embedding_size
segment_embeddings = n * embedding_size
position_embeddings = max_sequence_length * embedding_size
layer_norm = (embedding_size + embedding_size) * 3
layer_num = 4 * (embedding_size * embedding_size + embedding_size)
feed_num = embedding_size * hide_size + hide_size + hide_size * embedding_size + embedding_size
pool_num = embedding_size * embedding_size + embedding_size
num = token_embeddings + segment_embeddings + position_embeddings + layer_norm + layer_num + feed_num + pool_num
print(num)
print("模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
