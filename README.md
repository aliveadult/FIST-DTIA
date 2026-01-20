# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction

<img width="2587" height="1528" alt="FIST-DTIA Model Architecture" src="https://github.com/aliveadult/FIST-DTIA/raw/main/figure1.png" />

## 💡 FIST-DTIA Framework
[cite_start]**FIST-DTIA** (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) is a multi-modal deep learning framework designed to capture the complex "lock-and-key" mechanisms of molecular binding[cite: 8, 28]. [cite_start]By integrating 1D sequence data, 2D topological graphs, and 3D structural voxelization, FIST-DTIA provides robust dual-predictions for both drug–target interaction (DTI) and binding affinity (DTA)[cite: 11, 47].

### Core Architectural Innovations:
1. [cite_start]**Pharmacophore-Aware Drug Representation:** - Combines a GNN-based super-node aggregation mechanism to extract functional groups (pharmacophores) from 2D graphs[cite: 9, 54, 131].
   - [cite_start]Utilizes a 3D CNN to encode explicit spatial pharmacophore distributions via 7-channel voxel maps (H-Donor, H-Acceptor, Aromatic, etc.)[cite: 9, 150, 153].
2. [cite_start]**3D Swin Transformer for Protein Encoding:** - Employs a condensed 3D voxelization strategy (4 channels: density, hydrophobicity, and charge) for protein structures[cite: 10, 204, 237].
   - [cite_start]Captures global structural contexts and long-range dependencies using a **3D Swin Transformer** (W-MSA/SW-MSA), bypassing traditional all-atom computational bottlenecks[cite: 10, 52, 242].
3. [cite_start]**Dual-Prediction Module:** The framework hierarchically fuses features via Cross-Scale Attention and outputs through two dedicated heads[cite: 53, 72, 111]:
   - [cite_start]**DTA Regression Head:** Predicts continuous affinity values (e.g., $pK_d$, $pK_i$), optimized via MSE Loss[cite: 72, 270].
   - [cite_start]**DTI Classification Head:** Predicts binary interaction probability via a Softmax layer[cite: 72, 271].

---

## 🧠 Project Structure
The implementation is organized into modular components for structural abstraction and model training:

| File Name | Description |
| :--- | :--- |
| `mains.py` | [cite_start]Primary execution script for the **5-fold cross-validation** workflow and dual-task training/evaluation[cite: 308]. |
| `models.py` | [cite_start]Defines the `HGDDTI` architecture, featuring the `StructuralEncoder`, `3D Swin Transformer` modules, and the multi-modal fusion head[cite: 53]. |
| `utilss.py` | [cite_start]Core utilities for **3D Voxelization**, pharmacophore mapping, and graph construction with super-nodes[cite: 67, 148]. |
| `configss.py` | [cite_start]Configuration for hyperparameters (e.g., d_model: 256), voxel resolution ($32^3$), and data paths[cite: 151, 619]. |
| `evaluations.py` | [cite_start]Metric calculations for classification (Acc, AUC, AUPR) and regression (MSE, CI, $R_m^2$)[cite: 330, 331]. |

---

## 📁 Datasets & Evaluation
[cite_start]FIST-DTIA is rigorously validated across 12 benchmark datasets[cite: 338]:

* [cite_start]**DTI Classification:** Validated on DrugBank, Davis, KIBA, IC, E, BindingDB, BioSNAP, and Human datasets [cite: 283-291].
* [cite_start]**DTA Regression:** Quantitatively assessed on Davis, KIBA, Metz, and ToxCast benchmarks[cite: 295, 303].
* [cite_start]**Cold-Start Scenarios:** Proven robust in "Cold-drug" and "Cold-drug & target" settings, demonstrating superior generalization to novel chemical entities [cite: 315-322].

---

## 🛠️ Setup & Usage

### Installation
```bash
# Clone the repository
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)
cd FIST-DTIA

# Install dependencies
pip install torch torch-geometric rdkit biopython pandas tqdm
