# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:50:25 2021

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
    [1,0,1,1,0],
    [1,0,0,1,1],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [1,1,1,0,1],
]

df = pd.DataFrame(datasets,
                  columns=items,
                  index=users)
print(df)
##计算杰卡德相似度
A = jaccard_score(df['Item A'],df['Item B'])
##计算用户相似度
user_similar = 1-pairwise_distances(df.values,metric='jaccard')
user_similar = pd.DataFrame(user_similar,columns=users,index = users)

item_similar = 1-pairwise_distances(df.T.values,metric='jaccard')
item_similar = pd.DataFrame(item_similar,columns=items,index = items)

#为每一个用户找到最相似的2个用户
topN_users = {}
for i in user_similar.index:
    #取出每一列数据， 删除自己，按照相似度排序
    _df = user_similar.loc[i].drop([i])
    _df_sorted = _df.sort_values(ascending = False)
    top2 = list(_df_sorted.index[:2])
    topN_users[i] = top2


#根据topn的相似用户构建推荐结果
rs_results={}
for user,sim_users in topN_users.items():
    rs_result = set() #为每一用户保存推荐结果
    for sim_user in sim_users:
        rs_result = rs_result.union(set(df.ix[sim_user].replace(0,np.nan).dropna().index))
    #顾虑掉已经购买的商品
    rs_result -= set(df.ix[user].replace(0,np.nan).dropna().index)
    rs_results[user] = rs_result




















