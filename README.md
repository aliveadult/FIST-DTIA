# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction



## 💡 FIST-DTIA Framework
**FIST-DTIA** (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) 是一个旨在通过多维度特征融合捕捉分子结合机制的深度学习框架。该模型通过整合 1D 序列、2D 拓扑图和 3征 3D 空间表征，实现了对药物-靶标相互作用（DTI）和结合亲和力（DTA）的高精度双重预测 。

### 核心架构创新：
1. **药效团感知 3D 体素化 (Pharmacophore-Aware 3D Voxelization)**：
   - **配体表征**：结合 GNN 超节点聚合机制提取功能基团，并利用 3D CNN 通过体素图编码显式的药效团空间分布 。
2. **3D Swin Transformer 蛋白质表征**：
   - **高效抽象**：采用压缩的 3D 体素化策略处理蛋白质结构 。
   - **全局上下文**：利用 3D Swin Transformer 的 W-MSA 和 SW-MSA 机制捕捉蛋白质的全局结构信息和长程空间依赖关系 。
3. **双任务预测输出 (Dual-Prediction Output)**：
   - **DTA 回归头**：输出连续的亲和力数值（如 $pK_d$, $pK_i$），使用 MSE 损失函数进行优化 。
   - **DTI 分类头**：通过 Softmax 层输出相互作用的概率 。

---

## 🧠 项目结构与代码说明
根据代码库的实际实现，主要文件功能如下：

| 文件名 | 功能描述 |
| :--- | :--- |
| `mains.py` | 主执行程序，管理 **5 折交叉验证** 流程、模型初始化及训练/评估循环。 |
| `models.py` | 定义 `HGDDTI` 模型架构，包含 `StructuralEncoder` (GAT-based) 以及用于多模态融合的 `fusion_head`。 |
| `utilss.py` | 核心工具类，包含基于 **K-Means 的 3D 结构聚类**、药效团感知 2D 图构建及 PDB/SMILES 数据处理。 |
| `configss.py` | 全局配置文件，定义超参数（如 `d_model: 256`）、体素网格分辨率及数据路径。 |
| `evaluations.py` | 评估模块，计算分类（Acc, AUC, AUPR）与回归（MSE, CI, $R_m^2$）的各项指标。 |

---

## 📁 数据集
FIST-DTIA 在 12 个基准数据集上进行了广泛验证，涵盖了多种生物学场景：

**DTI 分类数据集**：包括 BindingDB (最大靶标多样性), DrugBank, BioSNAP 以及专门的离子通道 (IC) 和核受体 (E) 数据集 。
**DTA 回归数据集**：包括 Davis (激酶聚焦), KIBA, Metz 和大规模的 ToxCast 毒性亲和力数据集 。

---

## 🛠️ 环境搭建与使用

### 环境要求
- Linux (测试环境：CUDA 12.4)
- Python 3.9+
- NVIDIA GeForce RTX 3090/4090 (24G) 

### 安装步骤
```bash
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)
cd FIST-DTIA
pip install -r requirements.txt
# 主要依赖: torch, torch_geometric, rdkit, biopython, scikit-learn
