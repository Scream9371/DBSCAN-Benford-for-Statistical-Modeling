# DBSCAN Benford for Statistical Modeling Competition
本项目设计了一种结合DBSCAN聚类与Benford定律的DBSCAN-Benford算法，应用于在没有标签的财务欺诈数据集中标记高风险欺诈样本，为审查具有财务欺诈行为的公司提供决策参考。

以下是项目内容介绍：

1. `RESSET_QFFRAUDF_1.csv`：用于训练传统DBSCAN算法模型以及DBSCAN-Benford算法模型的数据集，包含各大上市公司2002年12月31日至2023年12月31日一年一次的总共60000条财务欺诈因子统计数据，统计字段为上市公司代码、最新公司全称、A股股票代码、B股股票代码、H股股票代码、上市标识、截止日期、DSRI、AQI、EPI、TATA、GMI、SGI、SGAI、LVGI以及对应的M打分和预测的财务操纵可能性等级。

2. `pre_des.py`：完成对原始数据集的预处理以及描述性统计工作。输出结果保存在 `result1` 目录中，内容包括：
   - `description_处理前统计结果.csv`：原始数据剔除缺失值后TATA、GMI、SGI、SGAI、DEPI五项指标的描述性统计信息；
   - `description_处理前箱线图.png`：剔除缺失值后TATA、GMI、SGI、SGAI、DEPI五项指标的箱线图；
   - `description_处理前直方图.png`：剔除缺失值后TATA、GMI、SGI、SGAI、DEPI五项指标的直方图；
   - `description_处理后箱线图.png`：IQR异常值处理后TATA、GMI、SGI、SGAI、DEPI五项指标的箱线图；
   - `description_处理后直方图.png`：IQR异常值处理后TATA、GMI、SGI、SGAI、DEPI五项指标的直方图；
   - `description_热力图.png`：IQR异常值处理后TATA、GMI、SGI、SGAI、DEPI五项指标之间的相关性热力图；
   - `qffraudf_clean.csv`：剔除缺失值后的完整财务欺诈数据。

3. `dbscan.py`：传统DBSCAN算法在本研究采用的数据集应用示例。输出结果保存在 `result2` 目录中，内容包括：
   - `dbscan_特征分布组合图.png`：TATA、GMI、SGAI、DEPI四项指标高风险组和低风险组对比的分布特征分布图；
   - `dbscan_特征重要性.png`：TATA、GMI、SGAI、DEPI四项指标高风险组和低风险组对比的重要性程度热力图；
   - `dbscan_聚类结果3D.png`：以GMI、SGAI、DEPI三项指标为例展示三维可视化的聚类结果；
   - `dbscan_异常检测结果.csv`：在 `qffraudf_clean.csv` 基础上添加高低风险标签；
   - `dbscan_低风险与高风险下的描述性统计.csv`：TATA、GMI、SGAI、DEPI四项指标分别在高风险组和低风险组下的描述性统计信息。

4. `Benford_factor.py`：计算各项指标的Benford因子。输出结果保存在 `result3` 目录中，内容包括：
   - `benford_检验结果.csv`：对选取的TATA、GMI、SGAI、DEPI四项指标进行卡方检验，判断是否满足Benford定律，若不满足，则进行Benford因子的构造；
   - `benford_频率对比.csv`：分别计算TATA、GMI、SGAI、DEPI四项指标的首位数字观测频率并与理论频率对比；
   - `benford_差异最大的首位数字.csv`：通过频率对比，得到每项指标首位数字观测频率并与理论频率差异最大的首位数字；
   - `benford_因子结果.csv`：对TATA、GMI、SGAI、DEPI四项指标生成Benford因子（异常样本标记为 1，正常样本为 0）。

5. `DBSCAN-Benford.py`：改进后的DBSCAN-Benford算法在本研究采用的数据集应用示例。输出结果保存在 `result4` 目录中，内容与 `dbscan.py` 的输出结果一致。
