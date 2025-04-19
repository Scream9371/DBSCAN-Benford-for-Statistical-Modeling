import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import os

# 设置全局字体为 SimHei 用于中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义输出目录
output_dir = os.path.join(os.getcwd(), "result2")
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 数据加载
input_file = os.path.join(os.getcwd(), "result1", "qffraudf_clean.csv")
df = pd.read_csv(input_file, nrows=60000)
features = [
    "应计系数()_TATA",
    "毛利率指数()_GMI",
    "销售管理费用指数()_SGAI",
    "折旧率指数()_DEPI",
]
X = df[features]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 参数自适应选择
def find_optimal_eps(X, k=4):
    """基于k-距离图的eps选择"""
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    return np.percentile(k_dist, 95)  # 取95%分位数


# 优化DBSCAN参数
optimal_eps = find_optimal_eps(X_scaled, k=5)  # 增加k值以获得更合适的eps
db = DBSCAN(eps=optimal_eps, min_samples=4).fit(X_scaled)  # 降低min_samples以获得更多簇

# 结果标记
df["cluster"] = np.where(db.labels_ == -1, -1, 0)  # 将所有非-1的簇标签统一设为0
n_clusters = 1  # 只保留一个低风险簇

# 计算轮廓系数（包含噪声点）
silhouette = silhouette_score(X_scaled, db.labels_)

# 箱线图和小提琴图组合
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
for idx, (feature, ax) in enumerate(zip(features, axes.ravel())):
    df["cluster_label"] = df["cluster"].map({-1: "高风险", 0: "低风险"})
    sns.violinplot(data=df, x="cluster_label", y=feature, ax=ax)
    sns.boxplot(
        data=df,
        x="cluster_label",
        y=feature,
        ax=ax,
        boxprops={"alpha": 0.5},
        showfliers=False,
    )
    ax.set_title(f'{feature.split("_")[1]}分布特征', fontsize=20, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "dbscan_特征分布组合图.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# 特征重要性热力图
feature_importance = np.zeros((len(features), 2))
df["cluster"] = np.where(df["cluster"] == -1, -1, 0)
# 计算特征重要性
for i, label in enumerate([-1, 0]):
    mask = df["cluster"] == label
    feature_importance[:, i] = np.abs(np.mean(X_scaled[mask], axis=0))

plt.figure(figsize=(10, 8))
sns.heatmap(
    feature_importance,
    xticklabels=["高风险", "低风险"],
    yticklabels=[f.split("_")[1] for f in features],
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
)
plt.savefig(
    os.path.join(output_dir, "dbscan_特征重要性.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# 3D聚类结果图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    X_scaled[:, 2],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.6,
)
ax.set_xlabel(features[0].split("_")[1])
ax.set_ylabel(features[1].split("_")[1])
ax.set_zlabel(features[2].split("_")[1])
plt.colorbar(scatter, label="聚类标签")
plt.savefig(
    os.path.join(output_dir, "dbscan_聚类结果3D.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# 保存统计结果到CSV文件
df.to_csv(
    os.path.join(output_dir, "dbscan_异常检测结果.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 计算每个簇的描述性统计量
descriptive_stats = []
for cluster in df["cluster"].unique():
    cluster_data = df[df["cluster"] == cluster]

    # 对每个特征计算统计量
    for feature in features:
        stats = {
            "簇": "高风险" if cluster == -1 else "低风险",
            "指标": feature.split("_")[1],
            "样本数": len(cluster_data),
            "均值": "{:.3f}".format(cluster_data[feature].mean()),
            "标准差": "{:.3f}".format(cluster_data[feature].std()),
            "最小值": "{:.3f}".format(cluster_data[feature].min()),
            "最大值": "{:.3f}".format(cluster_data[feature].max()),
            "中位数": "{:.3f}".format(cluster_data[feature].median()),
            "25分位数": "{:.3f}".format(cluster_data[feature].quantile(0.25)),
            "75分位数": "{:.3f}".format(cluster_data[feature].quantile(0.75)),
        }
        descriptive_stats.append(stats)

# 转换为DataFrame并保存
stats_df = pd.DataFrame(descriptive_stats)
stats_df.to_csv(
    os.path.join(output_dir, "dbscan_低风险与高风险下的描述性统计.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(
    f"""
=== 聚类结果分析 ===
* 检测到异常簇数量: {n_clusters}
* 噪声点占比: {(df.cluster == -1).mean():.1%}
* 轮廓系数: {f'{silhouette:.3f}' if silhouette is not None else '无法计算'}
"""
)
