import os
import torch
import numpy as np

class Configs:
    def __init__(self):
        # --- 通用设置 ---
        # 请根据你的实际路径修改 data_path
        self.data_path = '/media/6t/hanghuaibin/SaeGraphDTI/data/DAVIS/dataset.csv' 
        self.output_dir = 'output/hgddti_3d_swin_optimized/' 
        
        # ！！！关键修改 1：设置主设备为 cuda:0（DataParallel 的默认主设备）！！！
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # ！！！关键新增 2：指定使用的 GPU ID (GPU 0 和 GPU 1)！！！
        self.gpu_ids = [0, 1] if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else [0]
        
        # ！！！内存优化与配置修复！！！
        self.n_splits = 5             # K-Fold 交叉验证折数
        self.batch_size = 32           # 批处理大小减小 
        self.epochs = 260             # 训练轮次
        self.lr = 1e-3               # 学习率
        self.weight_decay = 1e-4      # 权重衰减
        self.random_state = 42 
        
        # 二分类阈值
        self.affinity_threshold = 7.0
        
        # --- ESM 嵌入设置 ---
        self.esm_embedding_path = '/media/6t/hanghuaibin/SaeGraphDTI/DAVIS_protein_esm_embeddings.pkl'
        self.protein_esm_dim = 1280 
        
        # ！！！新增：蛋白质 PDB 文件路径！！！
        self.pdb_dir = '/media/6t/hanghuaibin/SaeGraphDTI/data/3Ddata/prot_3d_for_DAVIS'
        
        # ！！！关键修改 3：蛋白质结构 3D Swin 参数 (瓶颈结构 C=128) ！！！
        self.voxel_size = 32             # 3D 体素图的边长 (从 48 调整到 32 缓解 OOM)
        self.swin_window_size = 4        # 3D Swin Window 大小
        self.swin_embed_dim = 128        # ！！！恢复为 128 维 (瓶颈维度)！！！
        self.swin_num_blocks = 2         # 3D Swin 块的数量 (偶数以保证 W-MSA/SW-MSA 交替)
        self.swin_in_channels = 4        # 多通道体素编码
        
        # --- 序列 Transformer 参数 ---
        self.d_model = 256          # 特征维度 (目标融合维度)
        self.nhead = 8
        self.num_transformer_layers = 6
        self.dropout = 0.4
        
        # 药物特征
        self.drug_fp_size = 1024       
        self.drug_vocab_size = self.drug_fp_size      
        self.drug_seq_len = self.drug_fp_size 
        
        # --- 图结构参数 (药物 GNN 保留) ---
        self.num_diffusion_steps = 6 
        self.num_heads_gat = 8       
        
        self.drug_node_dim = 32       # 药物原子特征维度
        self.protein_node_dim = 21    # 蛋白质残基特征维度 (已弃用)