# 实验二 · 基于 TuGraph 的 Gowalla 位置社交网络图分析

> 专业综合实践 · 作业 2
> 数据集：**Stanford SNAP `loc-Gowalla`**（196,591 用户 / 950,327 好友边 / 644 万签到）
> 图算法：**Node2Vec 节点嵌入**（链接预测 AUC **0.9118**）+ **PageRank 中心性**

---

## 一、任务与完成情况

| # | 任务 | 分值 | 完成情况 | 报告位置 |
|---|---|---|---|---|
| 1 | 选择数据集，描述来源、功能、结构 | 3 | ✅ Gowalla，实测 196,591 点 / 950,327 边 / 6,442,892 签到 | [2. 数据集](docs/实验二报告.md#二数据集) |
| 2 | 基于 TuGraph 实现图分析，≥2 种图算法且含节点嵌入 | 9 | ✅ Node2Vec + PageRank，17 张运行截图 | [5. Node2Vec](docs/实验二报告.md#五算法一node2vec-节点嵌入)、[6. PageRank](docs/实验二报告.md#六算法二pagerank-中心性) |
| 3 | 提交 CSDN / GitHub 链接 | 3 | ✅ 本仓库 | — |

📄 **完整实验报告：[`docs/实验二报告.md`](docs/实验二报告.md)**

---

## 二、核心结论

| 指标 | 结果 | 意义 |
|---|---|---|
| **Node2Vec 链接预测 AUC** | **0.9118** | 嵌入可支撑"可能认识的人"推荐 |
| PageRank 收敛 | 87 轮 | 幂迭代稳定 |
| **好友距离中位数** | **433 km**（随机对 6,435 km） | 复现 KDD 2011 结论：好友高度地理邻近 |
| 好友 `<100km` 占比 | **26.58%**（随机对 1.22%） | 推荐应优先本地 |
| Spearman(PageRank, 签到数) | 0.4718 | 社交结构与线下活跃是**部分独立**的维度 |
| 零签到但 PageRank 第 3 | 用户 505（5,420 好友） | **"潜水枢纽"存在**，按活跃度选种子会漏掉他 |

---

## 三、仓库结构

```
tugraph-assignment2/
├── README.md
├── docs/实验二报告.md          完整实验报告（含 17 张截图）
├── data/
│   ├── import.json             第一版导入配置（基础属性）
│   ├── import_v2.json          第二版导入配置（含算法结果）
│   ├── friend.csv              950,327 条去重无向边
│   └── user_base.csv           196,591 个用户基础特征
├── algorithms/
│   ├── node2vec.py             Node2Vec + 链接预测评估
│   ├── pagerank.py             PageRank + 相关性/地理/时间分析
│   ├── embedding_analysis.py   KMeans / t-SNE / 近邻 / 绘图
│   ├── merge_features.py       算法结果并回顶点表
│   └── out/                    全部指标 JSON 与图表
├── scripts/prep_gowalla.py     数据预处理
└── screenshots/                17 张运行截图与图表
```

---

## 四、图模型

**顶点 `user`（196,591）**

| 字段 | 类型 | 来源 |
|---|---|---|
| `id` | INT64 | 原始数据（主键） |
| `checkin_count` / `active_days` | INT32 | 签到聚合（时序特征） |
| `first_checkin` / `last_checkin` | STRING | 签到聚合（时间跨度） |
| `mean_lat` / `mean_lon` | DOUBLE | 签到聚合（活动重心） |
| `pagerank` / `degree` | DOUBLE / INT64 | **算法结果回写** |
| `cluster_id` | INT32 | **嵌入聚类回写** |
| `tsne_x` / `tsne_y` | DOUBLE | **t-SNE 坐标回写** |

**边 `friend`（950,327）**：`user → user`，无向，无属性，约束 `[["user","user"]]`。

> **建模取舍**：644 万条签到**不建顶点**，而是聚合成用户属性 —— 否则图会被大量二度叶子节点稀释，随机游走捕捉不到社区结构。128 万个地点同理（平均每点仅 5 次签到）。

---

## 五、算法一：Node2Vec 节点嵌入

**二阶有偏随机游走（p=1, q=2，偏 BFS）+ skip-gram**，5 条 × 40 步 × 196,591 节点 ≈ **3932 万步**，64 维，训练约 26 分钟。

**评估**：用 90% 边训练，10%（95,032 条）做链接预测，采样 10 万负例，按余弦相似度排序计算 AUC。

| 指标 | 值 |
|---|---|
| **AUC** | **0.9118** |
| 正例 / 负例平均相似度 | 0.6738 / 0.3245 |

t-SNE 可视化显示明显的**社区孤岛结构**；KMeans(k=8) 得到 8 个社群，簇 3 平均签到最高（48.79），簇 5 最低（15.99），可直接用于分群运营。

---

## 六、算法二：PageRank 中心性

无向图、阻尼系数 0.85、幂迭代 87 轮收敛。

| 排名 | 用户 | PageRank | 好友数 | 签到数 |
|---|---|---|---|---|
| 1 | 307 | 0.007163 | 14,730 | 225 |
| 2 | 220 | 0.007117 | 10,185 | 10 |
| 3 | 505 | 0.002928 | 5,420 | **0** |

**关键发现**：PageRank 衡量的是"社交结构位置"，不是使用强度。榜单第 3 的用户从未签到，却因社交枢纽地位影响力极高。

---

## 七、时间维度怎么用（回应教师意见）

1. **时序特征工程**：`active_days`（持续参与度）、`first/last_checkin`（生命周期），让"时间"进入向量空间；
2. **行为分布**：签到峰值在 UTC 17 点，月份峰值 2010-09 —— 但后者是平台增长趋势，**必须与个体行为区分**；
3. **相关性验证**：Spearman(PageRank, 签到数)=0.4718，说明线上社交与线下移动**部分独立**；
4. **地理时序结合**：好友距离中位数 433 km vs 随机 6,435 km，复现 KDD 2011 的"好友高度地理邻近"结论。

---

## 八、复现

```bash
python scripts/prep_gowalla.py                      # 预处理
lgraph_import --online true --online_type 1 -c data/import.json \
  -r http://127.0.0.1:7070 -u admin -p 73@TuGraph -g lab2_gowalla --overwrite true
python algorithms/node2vec.py --edges data/friend_edges.txt --out algorithms/out --walks 5 --length 40 --dim 64
python algorithms/pagerank.py --edges data/friend_edges.txt --users data/user.csv \
  --checkins data/loc-gowalla_totalCheckins.txt --out algorithms/out
python algorithms/embedding_analysis.py --out algorithms/out --users data/user.csv --k 8
```

依赖：`numpy<2`、`scipy`、`gensim`、`scikit-learn`、`matplotlib`（本实验安装在隔离目录以免污染系统环境）。

---

## 九、环境

| 项 | 值 |
|---|---|
| 图数据库 | TuGraph 4.5.2（WSL1 + Ubuntu 20.04，官方 deb 原生安装） |
| 子图 | `lab2_gowalla`（与实验一 `lab1_movie` 相互独立） |
| 导入方式 | `lgraph_import --online true --online_type 1`，**2.2 秒**完成百万级边 |
| 客户端 | Python 3.12 + numpy 1.26 + gensim 4.4 + scikit-learn 1.9 + matplotlib 3.11 |

---

## 十、参考资料

1. Gowalla 数据集：<https://snap.stanford.edu/data/loc-Gowalla.html>
2. Cho, Myers, Leskovec. *Friendship and Mobility: User Movement in Location-Based Social Networks*. KDD 2011
3. Grover & Leskovec. *node2vec: Scalable Feature Learning for Networks*. KDD 2016
4. Page, Brin, Motwani, Winograd. *The PageRank Citation Ranking*. 1999
5. TuGraph 数据导入工具：<https://tugraph-db.readthedocs.io/zh-cn/v4.2.0/6.utility-tools/1.data-import.html>
6. TuGraph 开源仓库：<https://github.com/TuGraph-family/tugraph-db>
7. Gensim Word2Vec：<https://radimrehurek.com/gensim/models/word2vec.html>
