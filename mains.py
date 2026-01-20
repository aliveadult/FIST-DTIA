import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
# 从 evaluations.py 导入回归指标计算函数
from evaluations import get_mean_and_std, calculate_regression_metrics 

# ----------------------------------------------------------------------
from utilss import HGDDTIDataset, collate_fn_combined, load_data, get_k_fold_data
# ----------------------------------------------------------------------

from models import HGDDTI
from configss import Configs

# 确保输出目录存在
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- 训练核心函数 (保持不变, 但损失函数改为 MSE) ---
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, affinity_batch, drug_pharma_voxel_batch in tqdm(data_loader, desc="Training"):
        
        if drug_graph_batch is None:
            continue

        # 将数据移动到主设备 (cuda:0)
        drug_graph_batch = drug_graph_batch.to(device)
        protein_voxel_batch = protein_voxel_batch.to(device) 
        drug_seq_batch = drug_seq_batch.to(device)
        protein_esm_batch = protein_esm_batch.to(device) 
        affinity_batch = affinity_batch.to(device)
        drug_pharma_voxel_batch = drug_pharma_voxel_batch.to(device) 

        optimizer.zero_grad()
        
        # output_prediction 是连续的亲和力预测值
        output_prediction = model(drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, drug_pharma_voxel_batch) 
        
        # 使用 MSELoss (回归任务)
        loss = criterion(output_prediction, affinity_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

# --- 评估核心函数 (修改为回归评估) ---

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    true_values = []
    pred_values = [] # 连续预测值
    
    with torch.no_grad():
        for drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, affinity_batch, drug_pharma_voxel_batch in tqdm(data_loader, desc="Evaluating"):
            
            if drug_graph_batch is None:
                continue

            drug_graph_batch = drug_graph_batch.to(device)
            protein_voxel_batch = protein_voxel_batch.to(device) 
            drug_seq_batch = drug_seq_batch.to(device)
            protein_esm_batch = protein_esm_batch.to(device) 
            affinity_batch = affinity_batch.to(device)
            drug_pharma_voxel_batch = drug_pharma_voxel_batch.to(device) 

            # output_prediction 是连续的亲和力预测值
            output_prediction = model(drug_graph_batch, protein_voxel_batch, drug_seq_batch, protein_esm_batch, drug_pharma_voxel_batch) 
            
            loss = criterion(output_prediction, affinity_batch)
            total_loss += loss.item()
            
            # 结果收集和指标计算在 CPU 上进行
            predictions = output_prediction.cpu().numpy().flatten()
            
            true_values.extend(affinity_batch.cpu().numpy().flatten())
            pred_values.extend(predictions)

    avg_loss = total_loss / len(data_loader)
    
    # 调用回归指标计算函数
    metrics = calculate_regression_metrics(np.array(true_values), np.array(pred_values))
    metrics['Loss'] = avg_loss # 这里的 Loss 实际是 MSE
    
    return metrics


# --- 主函数 (修改损失函数、优化指标和最佳模型选择标准) ---

def main():
    config = Configs()
    ensure_dir(config.output_dir)
    
    # 1. 设置主设备为 cuda:0
    device = torch.device(config.device)
    
    # 2. 加载数据
    df, esm_embeddings = load_data(config)
    
    # 3. K-Fold 划分
    k_folds = get_k_fold_data(df, config.n_splits, config.random_state)
    final_test_metrics = []
    
    drug_fp_size = config.drug_fp_size
    
    # 开始 K 折交叉验证循环
    for fold, (train_df, test_df) in enumerate(k_folds):
        print(f"\n==================== Starting Fold {fold + 1}/{config.n_splits} ====================")
        
        # 实例化模型和 DataParallel 包装
        raw_model = HGDDTI(drug_fp_size, config).to(device)
        
        if len(config.gpu_ids) > 1 and torch.cuda.device_count() >= len(config.gpu_ids):
            print(f"使用 {len(config.gpu_ids)} 个 GPU {config.gpu_ids} 进行 DataParallel 训练.")
            model = nn.DataParallel(raw_model, device_ids=config.gpu_ids)
        else:
            print(f"仅使用 {device}。")
            model = raw_model
            
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # 更改损失函数为均方误差 (MSELoss)
        criterion = nn.MSELoss() 
        
        train_dataset = HGDDTIDataset(train_df, esm_embeddings, config)
        test_dataset = HGDDTIDataset(test_df, esm_embeddings, config)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_combined, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_combined, num_workers=4)

        # 最佳模型标准：RMSE 越小越好
        best_rmse = float('inf') 
        best_epoch = 0
        
        # 训练循环
        for epoch in range(1, config.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_metrics = evaluate(model, test_loader, criterion, device)
            
            # !!! 打印新增的 R、CI 和 R2m 指标 !!!
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss (MSE): {test_metrics['Loss']:.4f} | "
                  f"RMSE: {test_metrics['RMSE']:.4f} | MAE: {test_metrics['MAE']:.4f} | "
                  f"R2: {test_metrics['R2']:.4f} | R: {test_metrics['R']:.4f} | CI: {test_metrics['CI']:.4f} | R2m: {test_metrics['R2m']:.4f}")

            # 根据 RMSE 分数保存最佳模型
            if test_metrics['RMSE'] < best_rmse:
                best_rmse = test_metrics['RMSE']
                best_epoch = epoch
                # 保存原始模型 state_dict
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, os.path.join(config.output_dir, f'best_model_fold_{fold+1}.pt'))

        
        # 载入最佳模型进行最终评估
        final_model = HGDDTI(drug_fp_size, config).to(device)
        final_model.load_state_dict(torch.load(os.path.join(config.output_dir, f'best_model_fold_{fold+1}.pt')))
        
        final_fold_metrics = evaluate(final_model, test_loader, criterion, device)
        final_test_metrics.append(final_fold_metrics)
        
        # !!! 打印新增的 R、CI 和 R2m 指标 !!!
        print(f"\nFold {fold + 1} Final Results (Epoch {best_epoch}): "
              f"RMSE: {final_fold_metrics['RMSE']:.4f}, MAE: {final_fold_metrics['MAE']:.4f}, "
              f"R2: {final_fold_metrics['R2']:.4f}, R: {final_fold_metrics['R']:.4f}, CI: {final_fold_metrics['CI']:.4f}, R2m: {final_fold_metrics['R2m']:.4f}")

    # ----------------------------------------------------------------------
    # 计算并打印 K-Fold 平均结果
    print("\n==================== K-Fold Cross-Validation Final Results ====================")
    # !!! 包含新增指标 !!!
    metric_names = ['RMSE', 'MAE', 'R2', 'R', 'CI', 'R2m'] 
    
    for name in metric_names:
        values = [m[name] for m in final_test_metrics]
        mean_val, std_val = get_mean_and_std(values)
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")
    # ----------------------------------------------------------------------


if __name__ == '__main__':
    main()