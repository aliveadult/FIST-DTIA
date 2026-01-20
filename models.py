import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch
from utilss import DRUG_PHARMACOPHORE_CHANNELS_SMARTS # 导入药效团配置

# --- 序列编码器 ---
class DrugSequenceEncoder(nn.Module):
    def __init__(self, fp_size, config):
        super(DrugSequenceEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(fp_size, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )

    def forward(self, seq_tokens):
        return self.proj(seq_tokens.float()) 

# --- 图结构编码器 (用于药物图 GNN) ---
class StructuralEncoder(nn.Module):
    def __init__(self, in_dim, config):
        super(StructuralEncoder, self).__init__()
        self.config = config
        
        self.conv1 = GATConv(in_dim, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        
        self.conv2 = GATConv(config.d_model, config.d_model // config.nhead, 
                             heads=config.nhead, dropout=config.dropout, concat=True)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        
        return h2 

# -----------------------------------------------------------
# 3D 窗口自注意力模块 (W-MSA)
# -----------------------------------------------------------

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: (B * nW, M^3, C)
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) 

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# -----------------------------------------------------------
# 3D 窗口操作辅助函数
# -----------------------------------------------------------

def window_partition_3d(x, window_size):
    """ 将输入特征图 (B, D, H, W, C) 划分为窗口 (B * nW, M^3, C) """
    B, D, H, W, C = x.shape
    
    if D % window_size != 0 or H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"D, H, W ({D},{H},{W}) 必须能被 window_size ({window_size}) 整除。")
        
    x = x.view(B, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size * window_size * window_size, C)
    return windows

def window_reverse_3d(windows, window_size, D, H, W):
    """ 将窗口 (B * nW, M^3, C) 反向重排为特征图 (B, D, H, W, C) """
    B_nW, M3, C = windows.shape
    num_windows = (D // window_size) * (H // window_size) * (W // window_size)
    B = B_nW // num_windows
    
    x = windows.view(B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, C)
    return x

# -----------------------------------------------------------
# Swin Transformer Block (交替 W-MSA 和 SW-MSA)
# -----------------------------------------------------------

class SwinTransformerBlock3D(nn.Module):
    """ 实现 W-MSA 和 SW-MSA 交替 """
    def __init__(self, dim, window_size, num_heads, dropout, shift_size):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        
        # 1. Shift Feature Map (仅对 SW-MSA 块执行)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # 2. Window Partition
        x_windows = window_partition_3d(shifted_x, self.window_size) 
        
        # 3. W-MSA / SW-MSA
        x_windows = self.attn(self.norm1(x_windows))
        
        # 4. Window Reverse
        shifted_x = window_reverse_3d(x_windows, self.window_size, D, H, W) 

        # 5. Reverse Shift (仅对 SW-MSA 块执行)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        
        # 6. MLP Block (带残差连接)
        x = shortcut + x 
        x = x + self.mlp(self.norm2(x))
        
        return x

# -----------------------------------------------------------
# 交叉注意力融合模块 (Interactive Attention)
# -----------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    使用交叉注意力融合药物结构特征和蛋白质结构特征。
    """
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim) 
        self.kv_proj = nn.Linear(dim, dim * 2) 

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, drug_feat, prot_feat):
        
        B, D = drug_feat.shape
        
        # 1. 投影 Q, K, V
        Q = self.query_proj(drug_feat).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        KV = self.kv_proj(prot_feat).reshape(B, 1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1] 

        # 2. 注意力计算
        Q = Q * self.scale
        attn = (Q @ K.transpose(-2, -1)) 
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 3. 加权求和
        context = (attn @ V).transpose(1, 2).reshape(B, D)
        
        # 4. 线性投影和残差连接
        fused_feat = drug_feat + self.proj_drop(self.proj(context))
        
        return fused_feat

# -----------------------------------------------------------
# 3D 结构编码器 (使用 SwinTransformerBlock3D)
# -----------------------------------------------------------

class ProteinStructuralEncoder(nn.Module):
    """ 堆叠 W-MSA 和 SW-MSA 块 """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 1. Patch Embedding: 嵌入层 (C_in -> C_embed)
        self.patch_embed = nn.Sequential(
             nn.Conv3d(config.swin_in_channels, config.swin_embed_dim, kernel_size=3, padding=1),
             nn.LayerNorm([config.swin_embed_dim, config.voxel_size, config.voxel_size, config.voxel_size])
        )

        # 2. Swin Blocks (W-MSA 和 SW-MSA 交替)
        shift_size = config.swin_window_size // 2 
        
        if config.swin_num_blocks % 2 != 0:
             print("警告: swin_num_blocks 必须是偶数以实现 W-MSA/SW-MSA 交替，已强制设置为偶数。")
             num_blocks = (config.swin_num_blocks // 2) * 2
        else:
             num_blocks = config.swin_num_blocks
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=config.swin_embed_dim,
                window_size=config.swin_window_size,
                num_heads=config.nhead,
                dropout=config.dropout,
                shift_size= 0 if i % 2 == 0 else shift_size # 偶数索引为 W-MSA (shift=0)，奇数索引为 SW-MSA (shift > 0)
            ) for i in range(num_blocks)
        ]) 

        # 3. 全局池化和投影
        self.global_pool = nn.AdaptiveAvgPool3d(1) 
        
        # 投影 (128 -> 256)
        self.final_proj = nn.Sequential(
            nn.Linear(config.swin_embed_dim, config.d_model * 2), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model) 
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        x = self.patch_embed(x) 
        x = x.permute(0, 2, 3, 4, 1).contiguous() 

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous() 
        
        # pooled_h 形状为 (B, 128)
        pooled_h = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1) 
        
        # final_vec 形状为 (B, 256)
        final_vec = self.final_proj(pooled_h) 
        
        return final_vec


# --- 药物药效团编码器 ---
class DrugPharmacophoreEncoder(nn.Module):
    """ 使用 3D CNN 对药物药效团体素图进行编码 """
    def __init__(self, in_channels, config):
        super().__init__()
        self.config = config
        
        # 使用 3D 卷积层对药效团体素图进行编码
        self.conv_blocks = nn.Sequential(
            # (B, C_in, V, V, V) -> (B, 64, V/2, V/2, V/2)
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # (B, 64, V/2, V/2, V/2) -> (B, 128, V/4, V/4, V/4)
            nn.Conv3d(64, config.d_model // 2, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm3d(config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 全局平均池化 (B, C_out, V/4, V/4, V/4) -> (B, C_out)
        self.global_pool = nn.AdaptiveAvgPool3d(1) 
        
        # 投影到 d_model 维度
        self.final_proj = nn.Linear(config.d_model // 2, config.d_model) # 128 -> 256

    def forward(self, x):
        # x 形状: (B, C_pharma, V, V, V)
        h = self.conv_blocks(x)
        h = self.global_pool(h).squeeze(-1).squeeze(-1).squeeze(-1) 
        final_vec = self.final_proj(h)
        return final_vec # 形状 (B, 256)


# --- 核心模型：HGDDTI (DTA 回归) ---
class HGDDTI(nn.Module):
    def __init__(self, drug_fp_size, config): 
        super(HGDDTI, self).__init__()
        self.config = config
             
        self.initial_atom_dim = 15 
        self.atom_proj = nn.Linear(self.initial_atom_dim, config.d_model)
        
        self.drug_seq_encoder = DrugSequenceEncoder(drug_fp_size, config)
        self.protein_esm_proj = nn.Linear(config.protein_esm_dim, config.d_model)
        self.drug_structural_encoder = StructuralEncoder(config.d_model, config)
        
        self.protein_structural_encoder = ProteinStructuralEncoder(config)
        
        self.drug_pharma_encoder = DrugPharmacophoreEncoder(
            in_channels=len(DRUG_PHARMACOPHORE_CHANNELS_SMARTS), # 5 个通道
            config=config
        )
        
        # 序列特征融合投影
        self.seq_proj = nn.Linear(config.d_model * 2, config.d_model)
        
        # 交叉注意力融合模块
        self.cross_attn_fusion = CrossAttentionFusion(
            dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout
        )
        
        # 最终回归头
        self.fusion_dim = config.d_model * 3 # Sequence (256) + Structure (256) + Pharmacophore (256) = 768
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, config.d_model), 
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1) # !!! 输出连续值，用于 DTA 回归 !!!
        )

    def forward(self, drug_graph_batch, protein_voxel_batch, drug_seq_data, protein_esm_vecs, drug_pharma_voxel_batch):
        
        current_device = self.atom_proj.weight.device
        
        # 1. 序列特征提取
        drug_seq_vec = self.drug_seq_encoder(drug_seq_data.to(current_device))       
        protein_seq_vec = F.relu(self.protein_esm_proj(protein_esm_vecs.to(current_device))) 
        
        # 2. 结构特征提取
        x_d = drug_graph_batch.x.to(current_device) 
        edge_index_d = drug_graph_batch.edge_index.to(current_device)
        x_d = F.relu(self.atom_proj(x_d)) 
        drug_structural_features_all = self.drug_structural_encoder(x_d, edge_index_d)
        
        protein_voxel_batch = protein_voxel_batch.to(current_device)
        protein_structural_vec = self.protein_structural_encoder(protein_voxel_batch) 

        # 提取药物药效团特征
        drug_pharma_voxel_batch = drug_pharma_voxel_batch.to(current_device)
        drug_pharma_vec = self.drug_pharma_encoder(drug_pharma_voxel_batch) 

        # 3. 提取药物超节点特征
        drug_structural_vecs = []
        start_idx = 0
        for i in range(drug_graph_batch.num_graphs):
            num_D = drug_graph_batch.num_drug_nodes[i].item() 
            super_node_index = start_idx + num_D 
            drug_structural_vecs.append(drug_structural_features_all[super_node_index])
            start_idx += (num_D + drug_graph_batch.num_super_nodes[i].item())
            
        drug_structural_vec = torch.stack(drug_structural_vecs, dim=0) 
        
        # 批次大小安全检查
        expected_batch_size = drug_seq_vec.size(0)
        if drug_structural_vec.size(0) != expected_batch_size:
             drug_structural_vec = drug_structural_vec[:expected_batch_size]

        # 4. 融合
        # 4a. 序列特征融合
        fused_seq = F.relu(self.seq_proj(torch.cat([drug_seq_vec, protein_seq_vec], dim=1)))
        
        # 4b. 结构特征交叉注意力融合
        fused_struct = self.cross_attn_fusion(drug_structural_vec, protein_structural_vec) 
        
        # 4c. 最终拼接和预测
        fused_features = torch.cat([fused_seq, fused_struct, drug_pharma_vec], dim=1) 
        
        # 输出连续的亲和力预测值
        output = self.fusion_head(fused_features)
        
        return output