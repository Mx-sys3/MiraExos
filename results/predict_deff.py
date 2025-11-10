import pandas as pd
import joblib
import numpy as np
import os

# 1. 设置文件路径
MODEL_PATH = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选/models-1/XGBoost.pkl"  # 之前保存的模型路径
FEATURE_NAMES_PATH = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选/models-1/XGBoost_feature.pkl"  # 特征名称路径
NEW_DATA_PATH = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/预测值/Collection.xlsx"  # 新数据文件路径
OUTPUT_PATH = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/预测值/Top10-prediction 10%.xlsx"  # 预测结果输出路径

# 2. 加载模型和特征名称
print("加载模型和特征名称...")
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
print(f"已加载模型: {type(model).__name__}")
print(f"模型使用的特征: {feature_names}")

# 3. 读取新数据
print(f"\n读取新数据: {NEW_DATA_PATH}")
new_df = pd.read_excel(NEW_DATA_PATH)

# 显示新数据的前几行
print("\n新数据预览:")
print(new_df.head())

# 4. 数据预处理（与训练数据保持一致）
print("\n进行数据预处理...")
processed_df = new_df.copy()

# 清理列名：去除首尾空格
processed_df.columns = processed_df.columns.str.strip()

# 创建更简洁的列名映射
col_map = {
    'Ethylene oxide (EO) units': 'EO_units'
}
processed_df = processed_df.rename(columns=col_map)

# 溶解度映射（如果存在Solubility列）
if 'Solubility' in processed_df.columns:
    solubility_map = {'极易溶': 5, '高度可溶': 4, '易溶': 3, '可溶': 2, '微溶': 1, '难溶': 0}
    processed_df['Solubility'] = processed_df['Solubility'].map(solubility_map)

# 5. 准备预测特征
print("\n准备预测特征...")
# 检查是否所有必需特征都存在
missing_features = [feat for feat in feature_names if feat not in processed_df.columns]
if missing_features:
    print(f"警告: 缺失以下特征: {missing_features}")
    print("请确保数据包含所有必需特征")
    # 退出或处理缺失特征
    exit(1)
else:
    print("所有必需特征都存在")

# 选择模型需要的特征
X_new = processed_df[feature_names]

# 显示特征数据
print("\n预测使用的特征数据:")
print(X_new.head())

# 6. 进行预测
print("\n进行预测...")
predictions = model.predict(X_new)
print(f"完成预测，共预测 {len(predictions)} 个化合物")

# 7. 创建结果数据框
results_df = processed_df.copy()
results_df['Predicted_Deff'] = predictions
results_df['Prediction_Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# 按预测值排序
results_df = results_df.sort_values('Predicted_Deff', ascending=False)

# 添加排名
results_df['Rank'] = range(1, len(results_df) + 1)

# 8. 保存结果
print(f"\n保存预测结果到: {OUTPUT_PATH}")
results_df.to_excel(OUTPUT_PATH, index=False)

# 9. 打印预测结果摘要
print("\n预测结果摘要:")
print(f"最高预测Deff: {results_df['Predicted_Deff'].max():.4f}")
print(f"最低预测Deff: {results_df['Predicted_Deff'].min():.4f}")
print(f"平均预测Deff: {results_df['Predicted_Deff'].mean():.4f}")

# 显示前5个化合物预测结果
print("\n预测结果最好的5个化合物:")
top_5 = results_df.head(5)[['Polymer candidates', 'Predicted_Deff', 'Rank']]
print(top_5)

print("\n预测完成! 结果已保存到指定文件")