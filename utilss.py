import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import os 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from configss import Configs
from rdkit import RDLogger 
import pickle 

# --- 新增 Biopython 导入和体素化 ---
from Bio.PDB import PDBParser
# ----------------------------------------

# 抑制 RDKit 警告
try:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR) 
except ImportError:
    pass

# 定义常用的药效团 SMARTS 模式（简化示例）
PHARMACOPHORE_SMARTS = {
    'Aromatic': '[a]',        
    'Donor_H': '[!#6!H0]-!@[#1]', 
    'Acceptor': '[!#6!H0;!#1;!H1]' 
}

### <<< NEW/MODIFIED >>> ###
# 药物药效团通道定义 (用于 3D 药效团特征图)
DRUG_PHARMACOPHORE_CHANNELS_SMARTS = {
    0: ('H_Donor', '[!#6!H0]-!@[#1]'),       # H 供体
    1: ('H_Acceptor', '[!#6!H0;!#1;!H1]'),    # H 受体
    2: ('Aromatic', '[a]'),                  # 芳香环
    3: ('Positive', '[+;!h1]'),             # 正电荷 (非氢原子)
    4: ('Negative', '[-;!h1]')              # 负电荷 (非氢原子)
}
### <<< NEW/MODIFIED >>> ###

# 简化残基化学属性映射 (基于 C-alpha)
# (1: 负电荷标志, 2: 疏水性标志)
RESIDUE_CHEMICAL_MAP = {
    # 负电荷 (酸性)
    'ASP': (-1, 0), 
    'GLU': (-1, 0),
    # 正电荷 (碱性)
    'ARG': (1, 0), 
    'LYS': (1, 0),
    'HIS': (1, 0),
    # 疏水性
    'ALA': (0, 1),
    'VAL': (0, 1),
    'LEU': (0, 1),
    'ILE': (0, 1),
    'MET': (0, 1),
    'PHE': (0, 1),
    'TYR': (0, 1),
    'TRP': (0, 1),
    # 极性/其他
    'SER': (0, 0),
    'THR': (0, 0),
    'CYS': (0, 0),
    'GLY': (0, 0), 
    'PRO': (0, 0),
    'ASN': (0, 0),
    'GLN': (0, 0),
    # 默认值
    'UNK': (0, 0),
}


def atom_features(atom):
    # ... (药物特征提取保持不变) ...
    allowable_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    atomic_num = atom.GetAtomicNum()
    if atomic_num in [6, 7, 8, 16, 15, 9, 17, 35, 53]:
        feature = np.zeros(10)
        feature[allowable_set.index(atom.GetSymbol())] = 1
    else:
        feature = np.array([0] * 9 + [1]) 
    
    additional_features = np.array([
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetDegree(),                          
        atom.GetImplicitValence(),                 
        int(atom.GetIsAromatic()),                 
        atom.GetFormalCharge()                     
    ])
    
    return np.concatenate([
        feature[:10], 
        additional_features
    ]).astype(np.float32)

def bond_features(bond):
    # ... (药物特征提取保持不变) ...
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE, 
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]).astype(int)


# --- ！！！修改：PDB 多通道体素化 (4 个通道) ！！！ ---
def voxelize_pdb(pdb_file_path, config):
    parser = PDBParser(QUIET=True)
    num_channels = config.swin_in_channels # 期望 4

    try:
        structure = parser.get_structure('protein', pdb_file_path)
    except Exception:
        # 返回 C=4 的零体素图
        return torch.zeros((num_channels, config.voxel_size, config.voxel_size, config.voxel_size), dtype=torch.float)

    # 提取所有 C-alpha 原子及其残基信息
    atom_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in RESIDUE_CHEMICAL_MAP:
                    try:
                        ca_atom = residue['CA']
                        resname = residue.get_resname()
                        atom_data.append({
                            'coord': ca_atom.get_coord(),
                            'prop': RESIDUE_CHEMICAL_MAP.get(resname, (0, 0)) # (Charge_flag, Hydrophobic_flag)
                        })
                    except KeyError:
                        continue 
    
    if not atom_data:
        return torch.zeros((num_channels, config.voxel_size, config.voxel_size, config.voxel_size), dtype=torch.float)
        
    coords = np.array([d['coord'] for d in atom_data])
    props = np.array([d['prop'] for d in atom_data]) # (N, 2)

    # 1. 坐标平移和缩放
    center = np.mean(coords, axis=0)
    max_coord = np.max(np.abs(coords - center))
    scale_factor = (config.voxel_size / 2.0) / (max_coord + 1e-6)
    coords_normalized = (coords - center) * scale_factor
    
    # 2. 映射到体素网格 [0, Voxel_size-1]
    voxel_coords = coords_normalized + (config.voxel_size / 2.0)
    voxel_coords = np.clip(voxel_coords, 0, config.voxel_size - 1).astype(int)
    
    # 3. 构建多通道体素图 (C x D x H x W)
    voxel_map = np.zeros((num_channels, config.voxel_size, config.voxel_size, config.voxel_size), dtype=np.float32)
    
    for i, (x, y, z) in enumerate(voxel_coords):
        charge_flag = props[i, 0]
        hydro_flag = props[i, 1]
        
        # 0: 密度 (Density)
        voxel_map[0, z, y, x] = 1.0 
        # 1: 负电荷 (Acidic/Negative)
        voxel_map[1, z, y, x] = np.maximum(0, -charge_flag) # 负电荷标志是 -1
        # 2: 正电荷 (Basic/Positive)
        voxel_map[2, z, y, x] = np.maximum(0, charge_flag)  # 正电荷标志是 1
        # 3: 疏水性 (Hydrophobic)
        voxel_map[3, z, y, x] = hydro_flag
        
    voxel_tensor = torch.tensor(voxel_map, dtype=torch.float) # 形状 (C, V, V, V)
    
    return voxel_tensor.unsqueeze(0).squeeze(0) # 保持 (C, V, V, V) 形状


### <<< NEW/MODIFIED >>> ###
def voxelize_drug_pharmacophore(mol, config):
    """
    生成药物的 3D 药效团特征体素图 (C_pharma, V, V, V)。
    """
    # 1. 确保分子有 3D 构象
    mol = Chem.AddHs(mol)
    try:
        # 尝试生成 3D 构象
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        coord = mol.GetConformer().GetPositions()
    except Exception:
        # 如果无法生成构象，返回零体素
        num_channels = len(DRUG_PHARMACOPHORE_CHANNELS_SMARTS)
        return torch.zeros((num_channels, config.voxel_size, config.voxel_size, config.voxel_size), dtype=torch.float)

    # 2. 坐标归一化/平移
    center = np.mean(coord, axis=0)
    max_coord = np.max(np.abs(coord - center))
    scale_factor = (config.voxel_size / 2.0) / (max_coord + 1e-6)
    coords_normalized = (coord - center) * scale_factor
    
    voxel_coords = coords_normalized + (config.voxel_size / 2.0)
    voxel_coords = np.clip(voxel_coords, 0, config.voxel_size - 1).astype(int)
    
    # 3. 构建多通道体素图
    num_channels = len(DRUG_PHARMACOPHORE_CHANNELS_SMARTS)
    voxel_map = np.zeros((num_channels, config.voxel_size, config.voxel_size, config.voxel_size), dtype=np.float32)

    for idx, (_, smarts) in DRUG_PHARMACOPHORE_CHANNELS_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None: continue
            
            # 获取匹配原子索引 (只取第一个原子作为代表，实际应用中可能需要更复杂的中心点计算)
            atom_indices_list = [match[0] for match in mol.GetSubstructMatches(patt)]
        except Exception:
             continue
        
        # 4. 映射特征到体素
        for atom_idx in atom_indices_list:
            if atom_idx < len(voxel_coords):
                x, y, z = voxel_coords[atom_idx]
                voxel_map[idx, z, y, x] = 1.0 # 在对应通道和体素位置标记
                
    voxel_tensor = torch.tensor(voxel_map, dtype=torch.float)
    return voxel_tensor.unsqueeze(0).squeeze(0) # 保持 (C, V, V, V) 形状
### <<< NEW/MODIFIED >>> ###


# --- 数据集类 (HGDDTIDataset) ---

class HGDDTIDataset(Dataset):
    def __init__(self, df, esm_embeddings, config):
        self.df = df
        self.config = config
        self.esm_embeddings = esm_embeddings 
        self.pdb_dir = config.pdb_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        drug_smiles = row['Drug']
        protein_sequence_key = row['Target Sequence'] 
        protein_name = row['Target_ID']

        try:
            affinity = float(row['Label']) 
        except ValueError:
            return None 
            
        # --- 1. 蛋白质 ESM 嵌入 (作为序列特征) ---
        if protein_sequence_key not in self.esm_embeddings:
             return None 
             
        full_esm_vec = torch.tensor(self.esm_embeddings[protein_sequence_key], dtype=torch.float)
        
        if full_esm_vec.dim() == 2 and full_esm_vec.size(1) == self.config.protein_esm_dim:
            protein_esm_vec = full_esm_vec.mean(dim=0) # 形状: (1280,)
        elif full_esm_vec.dim() == 1 and full_esm_vec.size(0) == self.config.protein_esm_dim:
            protein_esm_vec = full_esm_vec
        else:
            return None

        # --- 2. 药物结构 (图特征) ---
        mol = Chem.MolFromSmiles(drug_smiles)
        if mol is None: return None 

        atom_f = [atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_f: return None 
             
        x_d = torch.tensor(np.array(atom_f), dtype=torch.float)

        edge_index_d = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index_d.extend([(i, j), (j, i)])
        
        if not edge_index_d: edge_index_d = [(0, 0)] if x_d.size(0) > 0 else []

        edge_index_d = torch.tensor(edge_index_d, dtype=torch.long).t().contiguous()
        
        # --- 3. 药物超节点 ---
        x_s_d = x_d.mean(dim=0, keepdim=True)
        x_d_s = torch.cat([x_d, x_s_d], dim=0) 
        num_drug_nodes = x_d.size(0)
        super_node_index_d = num_drug_nodes
        
        pharma_atom_indices = set()
        for pattern_name, smarts in PHARMACOPHORE_SMARTS.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None: continue
                for match in mol.GetSubstructMatches(patt):
                    pharma_atom_indices.update(match)
            except Exception:
                continue
        
        nodes_to_connect = sorted(list(pharma_atom_indices)) if pharma_atom_indices else range(num_drug_nodes)

        new_edges = []
        for i in nodes_to_connect:
            new_edges.extend([(i, super_node_index_d), (super_node_index_d, i)])
        
        if new_edges:
             edge_index_new = torch.tensor(new_edges, dtype=torch.long).t()
             if edge_index_d.numel() == 0:
                 edge_index_d = edge_index_new
             else:
                 edge_index_d = torch.cat([edge_index_d, edge_index_new], dim=1)
             
        drug_data = Data(x=x_d_s, edge_index=edge_index_d, y=torch.tensor([affinity], dtype=torch.float))
        drug_data.num_drug_nodes = torch.tensor([num_drug_nodes], dtype=torch.long)
        drug_data.num_super_nodes = torch.tensor([1], dtype=torch.long) 
        drug_data.num_protein_block_nodes = torch.tensor([0], dtype=torch.long) 
        drug_data.num_protein_super_nodes = torch.tensor([0], dtype=torch.long) 

        # --- 4. 蛋白质结构 (3D 体素图) ---
        pdb_file = os.path.join(self.pdb_dir, f"{protein_name}.pdb")
        if not os.path.exists(pdb_file):
            # 返回 C=4 的零体素图
            protein_voxel_map = torch.zeros((self.config.swin_in_channels, self.config.voxel_size, self.config.voxel_size, self.config.voxel_size), dtype=torch.float)
        else:
             protein_voxel_map = voxelize_pdb(pdb_file, self.config)
             
        ### <<< NEW/MODIFIED >>> ###
        # --- 5. 药物 3D 药效团体素图 ---
        drug_pharma_voxel_map = voxelize_drug_pharmacophore(mol, self.config)
        ### <<< NEW/MODIFIED >>> ###
            
        
        # --- 6. 序列编码 ---
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.config.drug_fp_size)
        drug_token = torch.tensor([int(bit) for bit in mol_fp.ToBitString()], dtype=torch.float)

        ### <<< NEW/MODIFIED: 增加返回项 >>> ###
        return drug_data, protein_voxel_map, drug_token, protein_esm_vec, affinity, drug_pharma_voxel_map
        ### <<< NEW/MODIFIED: 增加返回项 >>> ###

def load_data(config):
    # ... (保持不变) ...
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"数据文件未找到: {config.data_path}")
    
    df = pd.read_csv(config.data_path)
    
    if 'Label' not in df.columns:
         raise KeyError("数据文件中必须包含 'Label' 列用于分类任务，但未找到。")
         
    try:
        with open(config.esm_embedding_path, 'rb') as f:
            esm_embeddings = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ESM 嵌入文件未找到: {config.esm_embedding_path}")
    except Exception as e:
        raise Exception(f"加载 ESM 嵌入文件时出错: {e}")
        
    return df, esm_embeddings 

def get_k_fold_data(df, n_splits, random_state):
    # ... (保持不变) ...
    labels = df['Label'].values
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_index, test_index in kf.split(df, labels): 
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        folds.append((train_df, test_df))
        
    return folds

### <<< NEW/MODIFIED: collate_fn_combined 增加处理药效团体素图 >>> ###
def collate_fn_combined(batch):
    batch = [item for item in batch if item is not None]
    # 原始返回: drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, affinity_batch 
    if not batch: return None, None, None, None, None, None # 增加一个 None 返回项

    drug_data_list = [item[0] for item in batch]
    protein_voxel_map_list = [item[1] for item in batch] # 3D Voxel Map 列表 (C, V, V, V)
    drug_token_list = [item[2] for item in batch]
    protein_esm_vec_list = [item[3] for item in batch] 
    affinity_list = [item[4] for item in batch]
    drug_pharma_voxel_map_list = [item[5] for item in batch] # 新增：药效团 3D Voxel Map 列表

    drug_graph_batch = Batch.from_data_list(drug_data_list)
    
    # 3D 体素图堆叠 (Batch_size, C, V, V, V)
    protein_voxel_batch = torch.stack(protein_voxel_map_list, dim=0) 
    drug_pharma_voxel_batch = torch.stack(drug_pharma_voxel_map_list, dim=0) # 新增堆叠
    
    drug_seq_batch = torch.stack(drug_token_list, dim=0) 
    protein_esm_batch = torch.stack(protein_esm_vec_list, dim=0) 
    
    affinity_batch = torch.tensor(affinity_list, dtype=torch.float).unsqueeze(1)
    
    # 增加 drug_pharma_voxel_batch 返回
    return drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, affinity_batch, drug_pharma_voxel_batch
### <<< NEW/MODIFIED: collate_fn_combined 增加处理药效团体素图 >>> ###