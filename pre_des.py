import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置全局字体为 SimHei 用于中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义保存目录
output_dir = os.path.join(os.getcwd(), "result1")
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取数据
file_path = os.getcwd() + r"\RESSET_QFFRAUDF_1.csv"
df = pd.read_csv(file_path, encoding="gbk", nrows=60000)

# 选择特征
features = [
    "应计系数()_TATA",
    "毛利率指数()_GMI",
    "营业收入指数()_SGI",
    "销售管理费用指数()_SGAI",
    "折旧率指数()_DEPI",
]

# 处理缺失值
missing_values = df[features].isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# 创建缺失值统计DataFrame
missing_stats = pd.DataFrame(
    {
        "指标名称": features,
        "缺失值数量": missing_values,
        "缺失值占比(%)": missing_percentage,
    }
)

# 添加清洗后样本数量信息
df_clean = df.dropna(subset=features)
clean_sample_count = len(df_clean)
missing_stats.loc["总计"] = ["清洗后样本数量", clean_sample_count, ""]

# 保存预处理数据到CSV文件
df_clean.to_csv(
    os.path.join(output_dir, "qffraudf_clean.csv"), index=False, encoding="utf-8-sig"
)

# 打印统计信息
print("\n各指标缺失值统计：")
print(missing_values)
print("\n缺失值占比：")
print(missing_percentage, "%")
print(f"\n删除缺失值后的样本数量: {len(df_clean)}")

# ============= 箱线图（处理前） =============
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_clean[features])
plt.ylabel("")
plt.xlabel("")
plt.xticks(range(len(features)), [f.split("_")[1] for f in features])
plt.yticks()
sns.despine()
plt.savefig(
    os.path.join(output_dir, "description_处理前箱线图.png"),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# ============= 直方图（处理前） =============
n_features = len(features)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    # 使用Freedman-Diaconis规则计算最优bin宽度
    data = df_clean[feature]
    iqr = data.quantile(0.75) - data.quantile(0.25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    bins = int((data.max() - data.min()) / bin_width)
    # 限制bins的数量在合理范围内
    bins = min(max(bins, 20), 100)

    sns.histplot(data, kde=True, ax=axes[i], bins=bins)
    axes[i].set_title(
        f'{feature.split("_")[1]} 分布直方图（异常值处理前）', fontsize=12
    )
    axes[i].set_xlabel("", fontsize=10)
    axes[i].set_ylabel("", fontsize=10)
    axes[i].tick_params(labelsize=8)

# 若子图数量为奇数，隐藏多余的子图
for j in range(n_features, n_rows * n_cols):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "description_处理前直方图.png"),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# 保存描述性统计结果（处理前）
description = df_clean[features].describe()
description.index = [
    "样本量",
    "均值",
    "标准差",
    "最小值",
    "25百分位数",
    "50百分位数",
    "75百分位数",
    "最大值",
]
description.columns = [f.split("_")[1] for f in features]
description.iloc[1:] = description.iloc[1:].applymap(lambda x: format(x, ".3f"))
description.to_csv(
    os.path.join(output_dir, "description_处理前统计结果.csv"), encoding="utf-8-sig"
)

# ========== 异常值处理（IQR 方法） ==========
for feature in features:
    Q1 = df_clean[feature].quantile(0.25)
    Q3 = df_clean[feature].quantile(0.75)
    IQR = Q3 - Q1
    df_clean[feature] = df_clean[feature].clip(
        lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR
    )

# ============= 箱线图（处理后） =============
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_clean[features])
plt.ylabel("")
plt.xlabel("")
plt.xticks(range(len(features)), [f.split("_")[1] for f in features])
plt.yticks()
sns.despine()
plt.savefig(
    os.path.join(output_dir, "description_处理后箱线图.png"),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# ============= 直方图（处理后） ===========
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    # 使用Freedman-Diaconis规则计算最优bin宽度
    data = df_clean[feature]
    iqr = data.quantile(0.75) - data.quantile(0.25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    bins = int((data.max() - data.min()) / bin_width)
    # 限制bins的数量在合理范围内
    bins = min(max(bins, 20), 100)

    sns.histplot(data, kde=True, ax=axes[i], bins=bins)
    axes[i].set_title(
        f'{feature.split("_")[1]} 分布直方图（异常值处理后）', fontsize=12
    )
    axes[i].set_xlabel("", fontsize=10)
    axes[i].set_ylabel("", fontsize=10)
    axes[i].tick_params(labelsize=8)

# 若子图数量为奇数，隐藏多余的子图
for j in range(n_features, n_rows * n_cols):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "description_处理后直方图.png"),
    dpi=600,
    bbox_inches="tight",
)
plt.close()

# ============= 热力图 =============
plt.figure(figsize=(10, 8))
simplified_labels = [f.split("_")[1] for f in features]
corr_matrix = df_clean[features].corr()
corr_matrix.index = simplified_labels
corr_matrix.columns = simplified_labels

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.xlabel("")
plt.ylabel("")
plt.xticks()
plt.yticks()
plt.savefig(
    os.path.join(output_dir, "description_热力图.png"), dpi=600, bbox_inches="tight"
)
plt.close()
