# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction



## 💡 FIST-DTIA Framework
FIST-DTIA (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) is a multimodal deep learning framework designed to capture the intricate landscape of molecular interactions. By integrating drug and protein features across 1D sequences and 3D structures, the model provides robust dual-predictions for drug discovery.

### Core Architectural Innovations
**Pharmacophore-Aware Drug Representation**
   - Combines a GNN-based super-node aggregation mechanism for functional group extraction.
   - Utilizes a 3D CNN to encode explicit spatial pharmacophore distributions via 7-channel voxel maps (H-Donor, Aromatic, Hydrophobic, etc.).
**3D Swin Transformer for Protein Encoding**
   - Employs a condensed 3D voxelization strategy (4 channels: density, hydrophobicity, charge) to bypass the computational bottlenecks of all-atom modeling.
   - Captures global structural contexts and long-range dependencies using alternating Window-based MSA (W-MSA) and Shifted-Window MSA (SW-MSA).
**Dual-Prediction Module**
   - The framework hierarchically fuses features via Cross-Scale Attention and outputs two distinct results:
     - **DTA Regression Head:** Predicts continuous binding affinity values (e.g., $pK_d$, $pK_i$).
     - **DTI Classification Head:** Predicts binary interaction probabilities.

---

## 🧠 Project Structure
The implementation is organized into modular components for data processing and model training:

| File Name | Description |
| :--- | :--- |
| `mains.py` | Primary execution file managing the K-Fold cross-validation, training loops, and metric reporting. |
| `models.py` | Defines the `HGDDTI` architecture, including 3D structural encoders and the dual-prediction fusion head. |
| `utilss.py` | Utility functions for K-Means structural clustering, 2D/3D graph construction, and pharmacophore mapping. |
| `configss.py` | Global configuration for hyperparameters, voxel dimensions, and ESM embedding settings. |
| `evaluations.py` | Functions for calculating classification (AUC, AUPR) and regression (MSE, CI) metrics. |

---

## 📁 Dataset & Evaluation
FIST-DTIA is evaluated on twelve diverse benchmark datasets:

### Quantitative & Qualitative Benchmarks
* **DTA Regression:** Davis ($pK_d$), KIBA (KIBA score), Metz ($pK_i$), and ToxCast ($AC_{50}$).
* **DTI Classification:** BindingDB, DrugBank, BioSNAP, Human, E (Enzymes), and IC (Ion Channels).

### Performance Highlights
* **SOTA Accuracy:** Outperforms existing baselines across all 12 benchmarks.
* **Generalization:** Maintains high stability in "Cold-drug" and "Cold-drug & target" scenarios.
* **Interpretability:** Validated through case studies on Type I (Erlotinib) and Allosteric (Trametinib) inhibitors.

---

## 🛠️ Environment Setup
Developed and tested on Linux with CUDA 12.4 using NVIDIA GeForce RTX 4090.

```bash
# 1. Clone the repository
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)

# 2. Install dependencies
pip install torch torch-geometric rdkit-pypi biopython scikit-learn pandas tqdm

# 3. Configure data paths in configss.py and run
python mains.py
