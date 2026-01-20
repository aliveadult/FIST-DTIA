# FIST-DTIA: Pharmacophore-Aware 3D Voxelization for Drug-Target Interaction and Affinity Dual-Prediction



## 🧪 Scientific Framework
[cite_start]FIST-DTIA (Pharmacophore-aware 3D Voxelization for Drug-target Interaction and Affinity) introduces a unified multimodal framework designed to address the "representational blindness" of sequence-only models in drug discovery[cite: 8, 30]. [cite_start]Unlike traditional approaches that rely on simplified 1D strings or 2D graphs, FIST-DTIA explicitly models the three-dimensional "lock-and-key" mechanism governing biological recognition[cite: 9, 28]. [cite_start]By synergizing chemical-identity-aware voxels with a global structural protein encoder, the model achieves high-fidelity molecular recognition across diverse therapeutic targets[cite: 12].

### Architectural Innovations
[cite_start]The framework is characterized by a five-branch encoding strategy that converges into a hierarchical fusion head[cite: 53]. [cite_start]Small molecules are represented through a pharmacophore-aware strategy that maps key functional groups into a 7-channel 3D dense grid, which is subsequently processed by a cross-channel 3D CNN to capture spatial chemical arrangements[cite: 147, 150]. [cite_start]Simultaneously, protein structures are abstracted into condensed 3D voxel maps and analyzed using a **3D Swin Transformer**[cite: 52]. [cite_start]This architecture utilizes alternating Window-based and Shifted-Window Multi-head Self-Attention to capture long-range spatial dependencies and allosteric signals, resolving the scalability constraints inherent to all-atom modeling[cite: 59, 233]. [cite_start]The final integration is managed by a **Cross-Scale Attention** module that fuses structural and sequence features (ESM and Morgan Fingerprints) before passing them to specialized heads for binary interaction classification and continuous affinity regression[cite: 70, 111].

---

## 📊 Benchmark Datasets and Resources
[cite_start]FIST-DTIA was rigorously validated on twelve public benchmarks, spanning varied species, target families (e.g., kinases, nuclear receptors, ion channels), and experimental measurement types[cite: 338, 339].

### Drug-Target Affinity (DTA) Regression
[cite_start]These datasets are used to evaluate the model's capacity to quantify binding strength via metrics such as Concordance Index (CI) and Mean Squared Error (MSE)[cite: 407].
* [cite_start]**Davis**: A kinase-focused dataset providing $pK_d$ values for 442 drugs and 68 targets[cite: 297, 299]. Available via [Therapeutics Data Commons](https://tdcommons.ai/multi_pred_tasks/dta/).
* [cite_start]**KIBA**: A large-scale benchmark that integrates different bioactivity types into a single normalized KIBA score[cite: 300]. Available via [TDC](https://tdcommons.ai/multi_pred_tasks/dta/).
* [cite_start]**Metz**: Features $pK_i$ values for over 35,000 pairs involving 170 targets[cite: 299, 301].
* [cite_start]**ToxCast**: A extensive toxicity dataset providing $AC_{50}$ values for over 300,000 chemical-protein interactions[cite: 303]. Accessible via [EPA ToxCast](https://www.epa.gov/chemical-research/toxicity-forecasting).

### Drug-Target Interaction (DTI) Classification
[cite_start]These benchmarks test the model's ability to discriminate between interacting and non-interacting pairs[cite: 282].
* [cite_start]**BindingDB**: A massive repository containing high target diversity (over 49,000 targets)[cite: 285, 293]. [Website](https://www.bindingdb.org/).
* [cite_start]**BioSNAP**: Derived from the Stanford Network Analysis Platform, providing human-centric chemical-gene interaction networks[cite: 290]. [Link](https://snap.stanford.edu/biodata/).
* [cite_start]**DrugBank**: Supplies clinically relevant balanced pairs of compounds and targets[cite: 284, 286]. [Database](https://go.drugbank.com/).
* [cite_start]**Specialized Sets (E, GPCR, IC)**: Nuclear receptors, G protein-coupled receptors, and Ion Channels derived from the Yamanishi "gold standard" benchmarks[cite: 291]. [Source](https://web.archive.org/web/20200218114639/http://web.kuicr.kyoto-u.ac.jp/supp/yamanishi/drugtarget/).

---

## 📂 Implementation Details
[cite_start]The project is structured for high-performance training on Linux-based GPU clusters[cite: 327].

| Component | Responsibility |
| :--- | :--- |
| **mains.py** | Orchestrates 5-fold cross-validation and manages the dual-task training pipeline. |
| **models.py** | Houses the `HGDDTI` class, defining the CNN/Transformer branches and the dual-head projection. |
| **utilss.py** | Handles **K-Means structural clustering** and 3D voxelization logic for ligands and proteins. |
| **configss.py** | Stores hyperparameters, including the 256-dimensional latent space and $32^3$ voxel grid resolution. |

---

## 🛠️ Usage Guide
FIST-DTIA requires Python 3.9+ and CUDA 11.8/12.4. 

```bash
# Clone the repository
git clone [https://github.com/aliveadult/FIST-DTIA.git](https://github.com/aliveadult/FIST-DTIA.git)
cd FIST-DTIA

# Install core dependencies
pip install torch torch-geometric rdkit-pypi biopython scikit-learn pandas tqdm

# Configuration: Update configss.py with local paths for ESM embeddings and PDB files
# Execute training and evaluation
python mains.py
