import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 用户-商品评分矩阵
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

# 计算商品之间的相似度
item_similarities = cosine_similarity(ratings.T)

# 对于用户0进行推荐
user_index = 0
user_ratings = ratings[user_index]

# 找到用户已经评分的商品
rated_items = user_ratings > 0

# 计算预测评分
predicted_ratings = item_similarities.dot(user_ratings)[rated_items] / np.array([np.abs(item_similarities).sum(axis=1)]).T[rated_items]

# 生成推荐列表
recommendations = np.argsort(predicted_ratings)[::-1]

print("推荐给用户{}的商品列表:".format(user_index))
for i in recommendations:
    if user_ratings[i] == 0:  # 只推荐用户未评分的商品
        print(i)