import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 读取数据
df = pd.read_csv("./df_clean.csv")

# 提取需要聚类的特征列，假设特征列为['feature1', 'feature2', ...]
# 请根据实际数据调整特征列的选择
features = df[['up_flow', 'down_flow', 'usage_time','hour']]

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

#设置DBSCAN的参数，例如eps和min_samples
eps = 2
min_samples = 5

# # 设置需要尝试的eps和min_samples值的范围
# param_grid = {
#     'eps': [0.2, 0.6,0.7,0.8,0.9,1,2,4],
#     'min_samples': [5, 10, 15, 20]
# }
#
# best_eps = None
# best_min_samples = None
# best_score = -1
#
# # 网格搜索
# for params in ParameterGrid(param_grid):
#     eps = params['eps']
#     min_samples = params['min_samples']
#
# 构造DBSCAN聚类器
estimator = DBSCAN(eps=eps, min_samples=min_samples)
#
 # 进行聚类
labels = estimator.fit_predict(scaled_data)
#
#     # 检查是否产生有效的簇（排除噪音点标签为-1的情况）
#     if len(np.unique(labels[labels != -1])) > 1:
#         # 计算轮廓系数，忽略噪音点
#         score = silhouette_score(scaled_data[labels != -1], labels[labels != -1])
#
#         if score > best_score:
#             best_score = score
#             best_eps = eps
#             best_min_samples = min_samples
#
# print(f"最佳eps值: {best_eps}")
# print(f"最佳min_samples值: {best_min_samples}")
# print(f"最佳轮廓系数: {best_score}")
#

# 统计每个簇中的样本数量
cluster_counts = np.bincount(labels + 1)

# 打印簇的数量（不包括噪音点，噪音点的标签为-1）
num_clusters = len(cluster_counts) - 1
print(f"聚类簇的数量为: {num_clusters}")

# 打印噪音点的数量
num_noise_points = np.sum(labels == -1)
print(f"噪音点的数量为: {num_noise_points}")

# 计算轮廓系数，忽略噪音点
silhouette_avg = silhouette_score(scaled_data[labels != -1], labels[labels != -1])
print(f"轮廓系数为: {silhouette_avg}")

# 计算每个簇的变量均值，忽略噪音点
cluster_means = df.groupby(labels).mean().iloc[:, :-1]

print("每个簇的变量均值:")
print(cluster_means)













