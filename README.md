# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction

<img width="2587" height="1528" alt="FIST-DTIA Framework" src="https://github.com/user-attachments/assets/你的图片链接" />

## 💡 FIST-DTIA Framework
**FIST-DTIA** (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) is a state-of-the-art multimodal deep learning framework designed for the dual-prediction of drug–target interaction (DTI) and binding affinity (DTA). 

By synergizing 1D sequence information with explicit 3D spatial representations, the model achieves high accuracy and robustness in molecular recognition tasks. The architecture centers on two primary innovations:

1. **Pharmacophore-Aware 3D Voxelization:**
   - **Ligand Representation:** Combines a GNN-based super-node aggregation mechanism to extract functional groups (pharmacophores) with a 3D CNN.
   - **Voxel Mapping:** Encodes explicit spatial pharmacophore distributions into 3D voxel maps, preserving the crucial geometric and chemical context often lost in 1D/2D models.

2. **3D Swin Transformer for Protein Representation:**
   - **Efficient Abstraction:** Utilizes a condensed 3D voxelization strategy for protein structures.
   - **Global Context:** Employs a 3D Swin Transformer to capture global structural contexts and long-range spatial dependencies, effectively bypassing the computational bottlenecks associated with large-scale atomic graph modeling.

### Key Contributions
* **Dual-Prediction Task:** Simultaneously optimizes for binary classification (DTI) and continuous regression (DTA) within a single framework.
* **Spatial Pharmacophore Encoding:** A novel strategy that maps functional groups into 3D space, enhancing the model's chemical interpretability.
* **Multimodal Fusion:** Hierarchically integrates 1D sequence embeddings (ESM, Morgan Fingerprints) with 3D structural voxel features via a hybrid MLP-Transformer network.
* **SOTA Performance:** Demonstrates superior predictive power across multiple benchmark datasets and proves highly effective in cold-start (unseen drugs/targets) scenarios.

---

## 🧠 Project Structure
The project is organized into modular components for feature extraction, 3D voxelization, and the training pipeline.

| File Name | Description |
| :--- | :--- |
| `config.py` | Global configuration for hyperparameters, voxel grid resolution, Swin Transformer settings, and data paths. |
| `voxel_utils.py` | Core utilities for **3D voxelization**, pharmacophore mapping, and 3D coordinate processing for both drugs and proteins. |
| `models.py` | Architecture definition of **FIST-DTIA**, including the 3D CNN voxel encoder, the 3D Swin Transformer, and the ESM/Fingerprint integration modules. |
| `dataset.py` | Custom data loaders for processing PDB structures, SMILES, and pre-computed ESM embeddings into multi-modal inputs. |
| `train_dual.py` | Main execution script for the dual-task training loop, implementing multi-task loss functions and performance monitoring. |
| `metrics.py` | Comprehensive evaluation suite for both classification (Acc, AUC, AUPR) and regression (RMSE, MSE, CI, $r^2$). |

---

## 📁 Dataset & Inputs
FIST-DTIA requires integrated sequence and structural data to perform 3D-aware predictions.

### Benchmark Datasets
The model has been rigorously validated on standard DTI/DTA benchmarks:
* **Davis & KIBA:** For affinity regression and interaction classification.
* **BindingDB:** Large-scale experimental binding data.
* **PDBbind:** High-quality 3D structural data for protein-ligand complexes.

### Required Inputs
1. **Protein PDB Files:** Used for 3D voxelization and Swin Transformer encoding.
2. **Ligand SMILES/SDF:** For generating Morgan fingerprints and 3D pharmacophore voxel maps.
3. **ESM Embeddings:** Pre-computed sequence features (e.g., ESM-2 or ESM-1b) for proteins.



---

## 🛠️ Environment Setup
Ensure you have a Linux environment with **CUDA 11.8/12.x** and an NVIDIA GPU (RTX 3090/4090 recommended).

```bash
# 1. Create and activate environment
conda create -n fist_dtia python=3.9
conda activate fist_dtia

# 2. Install PyTorch and 3D processing libraries
pip install torch torchvision torchaudio
pip install torch-geometric torch-sparse torch-cluster
pip install rdkit-pypi biopython scipy pandas tqdm

# 3. Install ESM for sequence embeddings
pip install fair-esm
