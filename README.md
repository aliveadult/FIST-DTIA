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
FIST-DTIA was rigorously validated on twelve public benchmarks, spanning varied species, target families (e.g., kinases, nuclear receptors, ion channels), and experimental measurement types.
### Drug-Target Affinity (DTA) Regression
These datasets evaluate the model's capacity to quantify binding strength via metrics such as Concordance Index (CI) and Mean Squared Error (MSE).
- **Davis**: A kinase-focused dataset providing $pK_d$ values for 442 drugs and 68 targets. It serves as a gold standard for testing regression fidelity. Available via [Therapeutics Data Commons](https://tdcommons.ai/multi_pred_tasks/dta/).
- **KIBA**: A large-scale benchmark that integrates different bioactivity types (IC50, Kd, Ki) into a single normalized KIBA score, offering a comprehensive view of binding profiles. Available via [TDC](https://tdcommons.ai/multi_pred_tasks/dta/).
- **Metz**: Includes $pK_i$ values for over 35,000 pairs involving 170 targets, providing a robust test for rank correlation and predictive stability.
- **ToxCast**: An extensive chemical toxicity dataset provided by the EPA, containing $AC_{50}$ values for over 300,000 interactions, representing a significant challenge for large-scale affinity prediction. Accessible via [EPA ToxCast](https://www.epa.gov/chemical-research/toxicity-forecasting).

### Drug-Target Interaction (DTI) Classification
These benchmarks assess the model's ability to discriminate between interacting and non-interacting pairs in binary scenarios.
- **BindingDB**: A massive repository containing high target diversity with over 49,000 targets, utilized to test the generalizability of molecular recognition. [Website](https://www.bindingdb.org/).
- **BioSNAP**: Collected from the Stanford Network Analysis Platform, it provides human-centric chemical-gene interaction networks for assessing clinically relevant pairs. [Link](https://snap.stanford.edu/biodata/).
- **DrugBank**: Supplies high-confidence balanced pairs of compounds and targets, essential for evaluating model robustness on clinically validated data. [Database](https://go.drugbank.com/).
- **Specialized Families (E, GPCR, IC)**: These "gold standard" benchmarks focus on Enzymes (E), G Protein-Coupled Receptors (GPCR), and Ion Channels (IC), providing a focused evaluation on specific protein folds.
---
## 📂 Implementation Architecture
The project is structured to ensure reproducibility and efficiency across GPU-accelerated environments.
- **mains.py**: Orchestrates the 5-fold cross-validation workflow and manages the dual-task training pipeline, ensuring balanced evaluation across regression and classification.
- **models.py**: Contains the primary model definition, housing the CNN/Transformer encoding branches and the dual-head projection modules.
- **utilss.py**: Handles the core geometric logic, including **K-Means structural clustering** and 3D voxelization for both ligands and proteins.
- **configss.py**: Stores all critical hyperparameters, such as the 256-dimensional latent space, learning rates, and the $32^3$ voxel grid resolution.

---

## 🛠️ Usage Guide
FIST-DTIA is designed for Python 3.9+ environments with CUDA support.
```bash
# Clone the repository
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)
cd FIST-DTIA

# Install dependencies
pip install torch torch-geometric rdkit-pypi biopython scikit-learn pandas tqdm
# Execute training and evaluation
# Ensure paths for ESM embeddings and PDB files are configured in configss.py
python mains.py
