import numpy as np
import pandas as pd
from filter_methods import *  # 导入filter_methods.py中的所有方法

# 设置您的CSV文件路径
csv_path = 'example02_1000_5.csv'  # 请将此替换为您的实际CSV文件路径

# 读取CSV文件，指定标题行
data = pd.read_csv(csv_path, header=0)

# 分离目标变量和特征
target = data.iloc[:, 0].values    # 第一列是目标变量
features = data.iloc[:, 1:].values  # 从第二列开始是特征

# **确保数据为数值型**
features = features.astype(np.float64)
target = target.astype(np.float64)

# **检查并处理缺失值（如果有）**
# 检查特征中的 NaN
if np.isnan(features).sum() > 0:
    print("特征中存在缺失值，正在填充缺失值...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
# 检查目标变量中的 NaN
if np.isnan(target).sum() > 0:
    print("目标变量中存在缺失值，正在填充缺失值...")
    target = np.nan_to_num(target, nan=np.nanmean(target))

# 选择要保留的前k个重要特征
topk = 5  # 您可以根据需要更改此值

# 应用filter_methods.py中的特征选择方法
# 不需要转置，因为特征已经是按列排列的
#result = Fisher_score(features, target)
result = SCC(features, target)

# 获取特征的排名
ranks = result.ranks  # 每个特征的排名数组
scores = result.scores  # 每个特征对应的得分

# 选择前k个特征的索引
topk_indices = np.argsort(ranks)[:topk]

# 如果您的数据有特征名称，获取前k个特征的名称
topk_feature_names = data.columns[1:][topk_indices]

# 简化后的特征矩阵
reduced_features = features[:, topk_indices]

# 可选：将目标变量重新添加到简化后的特征中
reduced_data = np.concatenate((np.expand_dims(target, axis=1), reduced_features), axis=1)

# 将简化后的数据集保存到新的CSV文件中
reduced_df = pd.DataFrame(reduced_data, columns=['meas'] + list(topk_feature_names))
reduced_df.to_csv('reduced_dataset.csv', index=False)

print("前k个重要特征是：")
print(topk_feature_names)
print("\n简化后的数据集已保存到 'reduced_dataset.csv'")
