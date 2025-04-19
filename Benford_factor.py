import pandas as pd
from scipy.stats import chisquare
import os

# 定义输出目录
output_dir = os.path.join(os.getcwd(), "result3")
os.makedirs(output_dir, exist_ok=True)


def construct_benford_factors(df, features, alpha=0.05):
    benford_law = {
        1: 0.301,
        2: 0.176,
        3: 0.125,
        4: 0.097,
        5: 0.079,
        6: 0.067,
        7: 0.058,
        8: 0.051,
        9: 0.046,
    }

    benford_factors = pd.DataFrame()
    frequency_data = []
    chi2_results = []
    max_diff_digits = []

    for col in features:
        # 过滤掉空值
        valid_data = df[col].dropna()
        # 提取首位数字（处理绝对值和零值）
        first_digits = valid_data.abs().apply(
            lambda x: int(
                str(x).lstrip("-0.").lstrip("0")[0]
                if str(x).lstrip("-0.").lstrip("0")
                and float(str(x).lstrip("-0.").lstrip("0")) > 0
                else 1
            )
        )
        # 重新索引，保持与原数据长度一致，缺失值用 NaN 填充
        first_digits = first_digits.reindex(df.index)

        # 计算观测频率
        observed = first_digits.value_counts(normalize=True).reindex(
            range(1, 10), fill_value=0
        )
        expected = pd.Series(benford_law)

        # 存储频率数据
        for d in range(1, 10):
            frequency_data.append(
                {
                    "指标": col.split("_")[1],
                    "首位数字": d,
                    "观测频率": round(observed[d], 3),
                    "理论频率": round(expected[d], 3),
                }
            )

        # 卡方检验
        chi2_stat, p_val = chisquare(observed * len(df), expected * len(df))
        compliance = p_val >= alpha

        # 存储检验结果
        chi2_results.append(
            {
                "指标": col.split("_")[1],
                "卡方统计量": f"{chi2_stat:.2f}",
                "P值": f"{p_val:.3f}",
                "显著性水平": alpha,
                "符合Benford律": "是" if compliance else "否",
            }
        )

        # 计算差异
        diff = (observed - expected).abs()
        max_diff_digit = diff.idxmax()
        max_diff_digits.append(
            {"指标": col.split("_")[1], "差异最大的首位数字": max_diff_digit}
        )

        # 生成Benford因子
        if not compliance:
            benford_factors[f"{col}_Benford"] = (first_digits == max_diff_digit).astype(
                int
            )

        # 添加每列的首位数字列
        benford_factors[f"{col}_首位数字"] = first_digits

    return (
        benford_factors,
        pd.DataFrame(frequency_data),
        pd.DataFrame(chi2_results),
        pd.DataFrame(max_diff_digits),
    )


# 读取数据
df = pd.read_csv(
    os.path.join(os.getcwd(), "result1", "qffraudf_clean.csv"), nrows=60000
)
features = [
    "应计系数()_TATA",
    "毛利率指数()_GMI",
    "销售管理费用指数()_SGAI",
    "折旧率指数()_DEPI",
]

# 执行分析
benford_factors, freq_df, chi2_df, max_diff_df = construct_benford_factors(df, features)
# 合并原始数据和 Benford 因子
combined_df = pd.concat([df, benford_factors], axis=1)

# 保存结果为CSV文件
combined_df.to_csv(
    os.path.join(output_dir, "benford_因子结果.csv"), index=False, encoding="utf-8-sig"
)
freq_df.to_csv(
    os.path.join(output_dir, "benford_频率对比.csv"), index=False, encoding="utf-8-sig"
)
chi2_df.to_csv(
    os.path.join(output_dir, "benford_检验结果.csv"), index=False, encoding="utf-8-sig"
)
max_diff_df.to_csv(
    os.path.join(output_dir, "benford_差异最大的首位数字.csv"),
    index=False,
    encoding="utf-8-sig",
)
print("成功保存CSV文件")
