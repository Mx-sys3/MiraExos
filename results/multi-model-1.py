import pandas as pd# type: ignore
import numpy as np# type: ignore
import matplotlib.pyplot as plt# type: ignore
import seaborn as sns# type: ignore
from matplotlib import rcParams# type: ignore
from sklearn.linear_model import LinearRegression# type: ignore
from sklearn.svm import SVR# type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor# type: ignore
from sklearn.neighbors import KNeighborsRegressor# type: ignore
from sklearn.model_selection import cross_val_predict, KFold# type: ignore
from sklearn.metrics import r2_score, mean_squared_error# type: ignore
from sklearn.inspection import permutation_importance# type: ignore
import xgboost as xgb# type: ignore

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据
df = pd.read_excel('D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/Data.20250804.xlsx', sheet_name='Sheet1')

# 打印实际列名以便调试
print("实际列名:", df.columns.tolist())

# 数据预处理
def preprocess_data(df):
    # 创建副本以避免修改原始数据
    df_processed = df.copy()
    
    # 清理列名：去除首尾空格
    df_processed.columns = df_processed.columns.str.strip()
    
    # 创建更简洁的列名映射
    col_map = {
        'Ethylene oxide (EO) units': 'EO_units',
        'Solubility': 'Solubility',  # 确保映射存在
    }
    df_processed = df_processed.rename(columns=col_map)
    
    # 检查并处理溶解度列
    if 'Solubility' in df_processed.columns:
        solubility_map = {'极易溶': 5, '高度可溶': 4, '易溶': 3, '可溶': 2, '微溶': 1, '难溶': 0}
        df_processed['Solubility'] = df_processed['Solubility'].map(solubility_map)
    else:
        print("警告: 'Solubility' 列不存在，跳过溶解度映射")
    
    # 特征选择
    features = ['Molecular weight', 'HLB', 'Solubility', 
                'EO_units', "Concentration"]
    
    # 检查所有特征是否存在
    available_features = [f for f in features if f in df_processed.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        print(f"警告: 以下特征不存在: {missing_features}")
        print(f"可用特征: {list(df_processed.columns)}")
    
    # 提取特征和目标变量 - 使用 'Predicted_Deff' 作为目标变量
    X = df_processed[available_features]
    y = df_processed['Predicted_Deff']  # 使用 'Predicted_Deff' 作为目标变量
    
    print("处理后的列名:", df_processed.columns.tolist())
    return X, y, df_processed

# 预处理数据
X, y, df_processed = preprocess_data(df)

# 模型选择 - 您可以根据需要修改这个值来选择不同的模型
MODEL_CHOICE = "xgb"  # 可选项: "lr", "svr", "gb", "knn", "rf","xgb"

# 根据选择初始化模型
if MODEL_CHOICE == "lr":
    model = LinearRegression()
    model_name = "线性回归(LR)"
elif MODEL_CHOICE == "svr":
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model_name = "支持向量回归(SVR)"
elif MODEL_CHOICE == "gb":
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, 
                                     max_depth=3, random_state=42)
    model_name = "梯度提升树(GBRT)"
elif MODEL_CHOICE == "knn":
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model_name = "K近邻回归(KNN)"
elif MODEL_CHOICE == "rf":
    model = RandomForestRegressor(n_estimators=200, max_depth=5, 
                                 min_samples_split=5, random_state=42)
    model_name = "随机森林(RF)"
elif MODEL_CHOICE == "xgb":
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                            n_estimators=200, 
                            learning_rate=0.1, 
                            max_depth=3, 
                            subsample=0.8,
                            random_state=42)
    model_name = "XGBoost"
else:
    raise ValueError("无效的模型选择")

print(f"\n使用模型: {model_name}")

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=kf)

# 评估指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n模型性能:")
print(f"RMSE = {rmse:.3f}")
print(f"R² = {r2:.3f}")

# 训练最终模型
model.fit(X, y)

# 获取特征重要性
if hasattr(model, 'feature_importances_'):
    # 树模型自带特征重要性
    importance = model.feature_importances_
elif hasattr(model, 'coef_'):
    # 线性模型使用系数的绝对值
    importance = np.abs(model.coef_)
else:
    # 其他模型使用排列重要性
    print("计算排列重要性...")
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance = result.importances_mean

feature_names = X.columns
sorted_idx = np.argsort(importance)[::-1]

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importance[sorted_idx], y=np.array(feature_names)[sorted_idx], palette='viridis')
plt.title(f'{model_name} - 特征重要性排序', fontsize=14)
plt.xlabel('相对重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()
plt.savefig(f'D:/AI/results/{model_name}_feature_importance.png', dpi=300)
plt.show()

# 创建特征重要性数据框
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': importance
}).sort_values('重要性', ascending=False)

# 打印特征重要性
print("\n特征重要性:")
print(importance_df)

# 预测并推荐聚合物
df_processed['预测_Deff'] = model.predict(X)
top_polymers = df_processed[['Polymer candidates', 'Predicted_Deff', '预测_Deff']].sort_values('预测_Deff', ascending=False).head(3)

print("\n推荐聚合物:")
print(top_polymers[['Polymer candidates', 'Predicted_Deff']])

# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际 Predicted_Deff', fontsize=12)
plt.ylabel('预测 Predicted_Deff', fontsize=12)
plt.title(f'{model_name} - 实际值 vs 预测值 (R² = {r2:.2f})', fontsize=14)
plt.grid(True, alpha=0.3)


# 标注样本点
if 'Polymer candidates' in df_processed.columns:
    for i, txt in enumerate(df_processed['Polymer candidates']):
        plt.annotate(txt, (y.iloc[i], y_pred[i]), textcoords="offset points", 
                     xytext=(0,5), ha='center', fontsize=8)
else:
    print("警告: 'Polymer candidates' 列不存在，无法添加标注")

plt.tight_layout()
plt.savefig(f'D:/AI/results/{model_name}_prediction_plot.png', dpi=300)
plt.show()

# 特征与目标变量的关系可视化
plt.figure(figsize=(14, 10))
num_features = len(feature_names)
rows = int(np.ceil(num_features / 3))  # 计算需要的行数

for i, feature in enumerate(feature_names):
    plt.subplot(rows, 3, i+1)
    sns.regplot(x=X[feature], y=y, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.xlabel(feature)
    plt.ylabel('Predicted_Deff')
    plt.title(f'{feature} vs Predicted_Deff')
plt.tight_layout()
plt.savefig(f'D:/AI/results/{model_name}_feature_relationships.png', dpi=300)
plt.show()

# ================== 新增代码：为三个树模型创建雷达图 ==================
if MODEL_CHOICE == "xgb":
    print("\n开始绘制树模型特征重要性雷达图...")
    
    # 只对树模型创建雷达图
    tree_models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', 
                                   n_estimators=200, 
                                   learning_rate=0.1, 
                                   max_depth=3, 
                                   subsample=0.8,
                                   random_state=42),
        "梯度提升树(GBRT)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, 
                                             max_depth=3, random_state=42),
        "随机森林(RF)": RandomForestRegressor(n_estimators=200, max_depth=5, 
                                       min_samples_split=5, random_state=42)
    }
    
    # 存储各模型的特征重要性
    model_importances = {}
    model_importances_raw = {}  # 存储原始重要性值
    
    # 训练模型并获取特征重要性
    for name, model in tree_models.items():
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            # 获取原始特征重要性
            imp_raw = model.feature_importances_
            model_importances_raw[name] = imp_raw
            
            # 归一化到0-1范围
            imp = (imp_raw - imp_raw.min()) / (imp_raw.max() - imp_raw.min())
            model_importances[name] = imp
        else:
            print(f"警告: {name} 没有特征重要性属性")
    
    # 打印特征重要性数据
    print("\n各模型特征重要性数据 (原始值):")
    importance_df = pd.DataFrame(model_importances_raw, index=feature_names)
    print(importance_df.round(4))
    
    print("\n各模型特征重要性数据 (归一化值):")
    importance_norm_df = pd.DataFrame(model_importances, index=feature_names)
    print(importance_norm_df.round(4))
    
    # 保存特征重要性数据到Excel
    with pd.ExcelWriter('D:/AI/results/tree_models_feature_importance.xlsx') as writer:
        importance_df.to_excel(writer, sheet_name='原始重要性')
        importance_norm_df.to_excel(writer, sheet_name='归一化重要性')
    print("特征重要性数据已保存到 'D:/AI/results/tree_models_feature_importance.xlsx'")
    
    # 准备雷达图数据
    labels = feature_names
    num_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 为每个模型绘制雷达图
    colors = ['#FF7F50', '#32CD32', '#4169E1']  # 珊瑚橙, 酸橙绿, 皇家蓝
    for i, (model_name, imp) in enumerate(model_importances.items()):
        # 归一化的重要性值
        values = imp.tolist()
        values += values[:1]  # 闭合多边形
        
        # 绘制雷达图
        ax.plot(angles, values, color=colors[i], linewidth=2, label=model_name)
    
    # 添加特征标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # 添加径向标签
    ax.set_rlabel_position(30)  # 将径向标签移到30度位置
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.1)
    
    # 添加标题和图例
    plt.title('树模型特征重要性比较 (归一化)', fontsize=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15), fontsize=12)
    
    # 添加网格
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('D:/AI/results/tree_models_feature_importance_radar.png', dpi=300)
    plt.savefig('D:/AI/results/tree_models_feature_importance_radar.pdf', dpi=300)
    plt.show()
    
    # 创建特征重要性排序表格
    print("\n各模型特征重要性排序:")
    for model_name, imp in model_importances_raw.items():
        sorted_idx = np.argsort(imp)[::-1]
        sorted_features = np.array(feature_names)[sorted_idx]
        sorted_importance = imp[sorted_idx]
        
        print(f"\n{model_name}:")
        for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importance)):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # 创建特征重要性对比柱状图
    plt.figure(figsize=(12, 8))
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, (model_name, imp) in enumerate(model_importances_raw.items()):
        plt.bar(x + i*width, imp, width, label=model_name)
    
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.title('树模型特征重要性对比', fontsize=14)
    plt.xticks(x + width, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:/AI/results/tree_models_feature_importance_bar.png', dpi=300)
    plt.show()

# 模型性能比较（可选）
# 当使用XGB模型时，运行所有模型并比较性能
# ============================================================
if MODEL_CHOICE == "xgb":
    print("\n开始模型性能比较...")
    
    # 定义所有模型
    models = {
        "线性回归(LR)": LinearRegression(),
        "支持向量回归(SVR)": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        "梯度提升树(GBRT)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
        "K近邻回归(KNN)": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "随机森林(RF)": RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=5, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        # 使用相同的交叉验证设置
        y_pred = cross_val_predict(model, X, y, cv=kf)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results.append({
            '模型': name,
            'R²': r2,
            'RMSE': rmse
        })
        print(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
    
    # 创建性能比较数据框
    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    
    # 可视化模型性能比较
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='R²', y='模型', data=results_df, palette='Blues_d')
    plt.title('模型R²分数比较', fontsize=14)
    plt.xlim(0, 1.0)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='RMSE', y='模型', data=results_df, palette='Reds_d')
    plt.title('模型RMSE比较', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/model_performance_comparison.png', dpi=300)
    plt.show()
    
    print("\n模型性能比较:")
    print(results_df)
    
    # 保存比较结果到Excel
    results_df.to_excel('D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/model_comparison_results.xlsx', index=False)
    print("模型比较结果已保存到 'D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/model_comparison_results.xlsx'")

    # ============================================================
    # 添加雷达图部分 - 比较XGBoost, GBRT和RF的特征重要性
    # ============================================================
    print("\n开始绘制特征重要性雷达图...")
    
    # 选择要比较的模型
    selected_models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, 
                                   learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42),
        "梯度提升树(GBRT)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, 
                                               max_depth=3, random_state=42),
        "随机森林(RF)": RandomForestRegressor(n_estimators=200, max_depth=5, 
                                         min_samples_split=5, random_state=42)
    }
    
    # 训练模型并获取特征重要性
    feature_importance_data = {}
    for name, model in selected_models.items():
        model.fit(X, y)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # 对于没有feature_importances_的模型，使用排列重要性
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance = result.importances_mean
        
        # 归一化特征重要性
        importance_normalized = importance / np.max(importance)
        feature_importance_data[name] = importance_normalized
    
    # 准备雷达图数据
    features = X.columns
    num_features = len(features)
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 为每个模型绘制雷达图
    colors = ['b', 'r', 'g']
    for i, (name, importance) in enumerate(feature_importance_data.items()):
        values = importance.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, color=colors[i], linewidth=2, label=name)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # 添加特征标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=12)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # 添加标题
    plt.title('模型特征重要性比较雷达图', fontsize=16, y=1.08)
    
    # 保存雷达图
    plt.tight_layout()
    plt.savefig('D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/feature_importance_radar.png', dpi=300)
    plt.show()
    
    # 创建特征重要性比较表格
    importance_comparison = pd.DataFrame(feature_importance_data, index=features)
    importance_comparison = importance_comparison.sort_values(by='XGBoost', ascending=False)
    
    print("\n特征重要性比较:")
    print(importance_comparison)
    
    # 保存特征重要性比较表格
    importance_comparison.to_excel('D:/AI/Polymer 筛选/20250730 CS-mExos@P/multi-model assay/feature_importance_comparison.xlsx')
    print("特征重要性比较表格已保存")