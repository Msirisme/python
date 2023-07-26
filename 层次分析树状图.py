import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

# 读取数据
ourData = pd.read_csv("./df_clean.csv")
ourData.head()

newData = ourData.iloc[:, [2, 3, 4, 5]].values
print(newData)

# 使用层次聚类算法找到最佳聚类数
dendrogram = sch.dendrogram(sch.linkage(newData, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# 根据树状图切割，选择聚类数量
num_clusters = 4
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(sch.linkage(newData, method='ward'), t=num_clusters, criterion='maxclust')

# 将聚类结果添加到原始数据集
ourData['cluster'] = clusters

# 计算每个簇的变量均值
cluster_means = ourData.groupby('cluster').mean()

# 输出带有分类和变量均值的原始数据集
print("带有分类和变量均值的原始数据集：")
print(ourData)

# 输出每个簇的变量均值
print("每个簇的变量均值：")
print(cluster_means)
