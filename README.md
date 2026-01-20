# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction

![Architecture](https://github.com/aliveadult/FIST-DTIA/blob/main/SwinBlock%20and%20pharmacophore%20graph%20convolution%20combined%20with%20drug-protein%20mixed%20attention)
*Figure: The FIST-DTIA framework integrates 1D sequence data, 2D pharmacophore-aware graphs, and 3D structural voxels via a Swin Transformer and Cross-Attention for high-fidelity prediction.*

## 🧪 Scientific Framework
FIST-DTIA (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) introduces a unified multimodal framework designed to address the representational limitations of sequence-only models. Unlike traditional approaches, FIST-DTIA explicitly models the three-dimensional "lock-and-key" mechanism by synergizing 3D voxelized pharmacophores with a global structural protein encoder.

### Key Innovations in Architecture:
* **3D Drug Pharmacophore Encoder**: Beyond 2D graphs, the model voxelizes drugs into a **7-channel 3D grid** (H-Donor, Acceptor, Aromatic, Positive, Negative, Hydrophobic, and Halogen). A 3D CNN extracts spatial chemical signatures from these voxels.
* **3D Swin Transformer Protein Encoder**: Processes protein structures via a **4-channel voxel map** (Density, Acidic, Basic, Hydrophobic). It utilizes alternating **Window-based (W-MSA)** and **Shifted-Window (SW-MSA)** self-attention to capture long-range spatial dependencies.
* **Interactive Cross-Attention Fusion**: Instead of simple concatenation, a **Cross-Attention module** allows the drug's structural features to "query" the protein's structural environment, facilitating adaptive feature integration.
* **Hybrid Representation**: Combines 1D sequence features (ESM-2 for proteins, Morgan Fingerprints for drugs) with 2D Graph Neural Networks (GAT) and 3D Voxel encoders.

---

## 📊 Benchmark Datasets and Resources
FIST-DTIA is designed to be validated on major DTA and DTI benchmarks.

### Drug-Target Affinity (DTA) & Interaction (DTI)
* **KIBA & Davis**: Standard benchmarks for affinity regression ($pK_d$ and KIBA scores).
* **BindingDB & BioSNAP**: Large-scale interaction datasets for binary classification.
* **ToxCast**: Evaluates chemical-biological interactions across thousands of targets.
* **Specialized Families**: Gold standard datasets for Enzymes (E), GPCRs, and Ion Channels (IC).

---

## 📂 Implementation Architecture
The project is modularized for clarity and high-performance computing:

* **`mains.py`**: Orchestrates the **5-Fold Cross-Validation** workflow. It includes a robust training/evaluation loop with **PyTorch DataParallel** support for multi-GPU acceleration.
* **`models.py`**: Defines the `HGDDTI` architecture, including the `ProteinStructuralEncoder` (Swin-3D), `DrugPharmacophoreEncoder` (CNN-3D), and `CrossAttentionFusion`.
* **`utilss.py`**: Contains the geometry engine. It handles RDKit-based **3D pharmacophore voxelization** for drugs and Biopython-based **C-alpha voxelization** for proteins.
* **`configss.py`**: Centralized hyperparameter management (e.g., $32^3$ voxel size, $128$ Swin embedding dimension, $256$ latent fusion dimension).
* **`evaluations.py`**: Utility functions for classification metrics (AUC, AUPR, F1) and statistical analysis.

---

## 🛠️ Usage Guide
FIST-DTIA requires a Python 3.9+ environment with CUDA support.

### 1. Installation
```bash
# Clone the repository
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)
cd FIST-DTIA

# Install dependencies
pip install torch torch-geometric rdkit-pypi biopython scikit-learn pandas tqdm
