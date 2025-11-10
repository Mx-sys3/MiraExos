import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取10%浓度数据以获取坐标轴范围
data_path_10 = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选/10%.xlsx"
df_10 = pd.read_excel(data_path_10)

# 计算10%浓度数据的坐标轴范围
eo_min_10 = df_10['EO_units'].min()
eo_max_10 = df_10['EO_units'].max()
hlb_min_10 = df_10['HLB'].min()
hlb_max_10 = df_10['HLB'].max()
deff_min_10 = df_10['Predicted_Deff'].min()
deff_max_10 = df_10['Predicted_Deff'].max()

print(f"10%浓度数据范围:")
print(f"EO范围: [{eo_min_10:.2f}, {eo_max_10:.2f}]")
print(f"HLB范围: [{hlb_min_10:.2f}, {hlb_max_10:.2f}]")
print(f"Deff范围: [{deff_min_10:.4f}, {deff_max_10:.4f}]")

# 使用10%浓度的坐标轴范围
EO_RANGE = (eo_min_10, eo_max_10)
HLB_RANGE = (hlb_min_10, hlb_max_10)
DEFF_RANGE = (deff_min_10, deff_max_10)

# 读取5%浓度数据
data_path = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选/5%.xlsx"
df = pd.read_excel(data_path)

# 检查浓度值
concentrations = df['Concentration'].unique()
print(f"5%数据中的浓度值: {concentrations}")

# 为每个浓度创建散点图
for concentration in concentrations:
    print(f"\n处理浓度: {concentration}")
    
    # 筛选该浓度的数据
    df_concentration = df[df['Concentration'] == concentration]
    
    # 创建散点图
    plt.figure(figsize=(12, 8))
    
    # 使用散点图 - 使用10%浓度的范围
    scatter = plt.scatter(
        df_concentration['EO_units'], 
        df_concentration['HLB'], 
        c=df_concentration['Predicted_Deff'], 
        cmap='rainbow', 
        alpha=0.7,
        s=20,
        vmin=DEFF_RANGE[0],  # 使用10%浓度的颜色映射范围
        vmax=DEFF_RANGE[1]   # 使用10%浓度的颜色映射范围
    )
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Predicted Deff', fontsize=12)
    
    # 设置坐标轴范围 - 使用10%浓度的范围
    plt.xlim(EO_RANGE[0], EO_RANGE[1])
    plt.ylim(HLB_RANGE[0], HLB_RANGE[1])
    
    # 设置标题和轴标签
    plt.xlabel('Ethylene Oxide (EO) Units', fontsize=12)
    plt.ylabel('HLB', fontsize=12)
    plt.title(f'Virtual Polymers - Concentration {concentration}\n(EO vs HLB colored by Predicted Deff)', fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    output_dir = "D:/AI/Polymer 筛选/20250730 CS-mExos@P/虚拟筛选"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"scatter_plot_Concentration_{concentration}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"散点图已保存到: {output_path}")
    
    # 显示图像
    plt.show()
    
    # 打印统计信息
    print(f"数据点数量: {len(df_concentration)}")
    print(f"Deff 最小值: {df_concentration['Predicted_Deff'].min():.4f}")
    print(f"Deff 最大值: {df_concentration['Predicted_Deff'].max():.4f}")
    print(f"Deff 平均值: {df_concentration['Predicted_Deff'].mean():.4f}")