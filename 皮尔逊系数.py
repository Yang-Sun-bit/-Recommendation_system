# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:47:17 2021

@author: Lenovo
"""
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_distances
users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 用户购买记录数据集
datasets = [
    [5,3,4,4,None],
    [3,1,2,3,3],
    [4,3,4,3,5],
    [3,3,1,5,4],
    [1,5,5,2,1],
]

df = pd.DataFrame(datasets,
                  columns=items,
                  index=users)

print("用户之间的两两相似度：")
# 直接计算皮尔逊相关系数
# 默认是按列进行计算，因此如果计算用户间的相似度，当前需要进行转置
user_similar = df.T.corr()
print(user_similar.round(4))

print("物品之间的两两相似度：")
item_similar = df.corr()
print(item_similar.round(4))

dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
ratings = pd.read_csv('ml-latest-small/ratings.csv',dtype=dtype,usecols = range(3))

ratings.head()
#构建透视表 找到 用户和电影之间的评分关系
ratings_matrix = ratings.pivot_table(index = ['userId'],columns=['movieId'],values='rating')










