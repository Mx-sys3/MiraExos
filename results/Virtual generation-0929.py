import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 创建保存目录（如果不存在）
save_dir = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选"
os.makedirs(save_dir, exist_ok=True)

# 原始数据
data = {
    "Polymer candidates": ["T701", "T707", "T803", "T908", "T1107", "T1301", "T1304", "T1307", "T1508", "T150R1", 
                          "P181", "P184", "P237", "P238", "P124", "P401", "P403", "P84", "P85", "P335", 
                          "P10R5", "P101", "P235", "P122", "P183", "P333", "P234", "P185", "P215", "P123", 
                          "P125", "P212", "P334", "P402", "P182", "Tween 20", "Tween 40", "Tween 60", "Tween 80", "Tween 85"],
    "Polarity": ["Amphiphilic"] * 40,
    "Molecular weight": [3600, 12500, 5500, 25000, 15000, 6800, 10500, 18000, 29900, 7900,
                         2000, 2900, 7835, 11400, 2225, 4400, 5800, 4200, 4600, 6500,
                         2000, 1100, 4600, 1630, 2650, 4950, 4200, 3400, 4150, 1850,
                         2400, 2750, 5900, 5000, 2500, 1226, 840, 609, 429, 428.6],
    "HLB": [4, 27, 15, 24, 21, 4, 15, 24, 24, 4, 4, 13, 24, 24, 4.5, 1, 8, 16.5, 15, 15,
            15, 3.5, 15, 4.3, 13, 15.4, 16.5, 15, 15, 4.2, 5.2, 9.5, 14.8, 14, 5.8, 
            16.7, 15.6, 14.9, 15, 11],
    "Solubility": ["微溶", "微溶", "易溶", "易溶", "高度可溶", "极易溶", "可溶", "高度可溶", "难溶", "难溶",
                   "难溶", "可溶", "易溶", "易溶", "可溶", "可溶", "可溶", "难溶", "难溶", "易溶",
                   "易溶", "易溶", "可溶", "可溶", "可溶", "可溶", "可溶", "可溶", "可溶", "可溶",
                   "可溶", "可溶", "可溶", "可溶", "可溶", "极易溶", "易溶", "可溶", "可溶", "微溶"],
    "Ethylene oxide (EO) units": [8.4, 216, 44, 456, 240, 16, 85.6, 288, 516, 20, 2, 26, 128, 208, 24,
                                  8, 40, 38, 50, 148, 22, 4, 71.8, 18, 28, 85, 38, 34, 40, 16, 21,
                                  26, 68, 56, 22, 20, 20, 20, 20, 20]
}

df = pd.DataFrame(data)

# 准备训练数据
X_mw = df[["HLB", "Ethylene oxide (EO) units"]]
y_mw = df["Molecular weight"]

# 训练分子量预测模型
mw_model = RandomForestRegressor(n_estimators=100, random_state=42)
mw_model.fit(X_mw, y_mw)

# 准备溶解度分类数据
le = LabelEncoder()
y_sol = le.fit_transform(df["Solubility"])

# 训练溶解度预测模型
sol_model = RandomForestClassifier(n_estimators=100, random_state=42)
sol_model.fit(X_mw, y_sol)

# 生成10000条虚拟数据
np.random.seed(42)
n_samples = 10000

# 生成离散均匀分布的HLB和EO
hlb_bins = np.linspace(0, 30, 301)  # 0.1间隔
eo_bins = np.linspace(0, 100, 1001)  # 0.1间隔

hlb_values = np.random.choice(hlb_bins, size=n_samples)
eo_values = np.random.choice(eo_bins, size=n_samples)

# 创建虚拟数据DataFrame
virtual_df = pd.DataFrame({
    "HLB": hlb_values,
    "Ethylene oxide (EO) units": eo_values
})

# 预测分子量
virtual_df["Molecular weight"] = mw_model.predict(virtual_df[["HLB", "Ethylene oxide (EO) units"]])
virtual_df["Molecular weight"] = virtual_df["Molecular weight"].round().astype(int)

# 预测溶解度
sol_pred = sol_model.predict(virtual_df[["HLB", "Ethylene oxide (EO) units"]])
virtual_df["Solubility"] = le.inverse_transform(sol_pred)

# 添加聚合物名称和极性
virtual_df["Polymer candidates"] = ["VC_" + str(i).zfill(6) for i in range(1, n_samples+1)]
virtual_df["Polarity"] = "Amphiphilic"

# 重新排列列顺序
virtual_df = virtual_df[["Polymer candidates", "Polarity", "Molecular weight", "HLB", 
                         "Solubility", "Ethylene oxide (EO) units"]]

# 保存到Excel - 修改为指定路径
save_path = os.path.join(save_dir, "virtual_polymers.xlsx")
virtual_df.to_excel(save_path, index=False)

print(f"虚拟化合物数据已生成并保存到 {save_path}")
print(f"生成数据示例:\n{virtual_df.head()}")