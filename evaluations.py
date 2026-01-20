import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 新增 Concordance Index (C-index) 计算函数 ---
def concordance_index(y_true, y_pred):
    """Calculates the Concordance Index (C-index) for DTA regression."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_pairs = 0
    c_index = 0
    
    # 迭代所有唯一的配对 (i, j)，其中 i < j
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            
            # 检查真实值是否非平局
            if y_true[i] != y_true[j]:
                n_pairs += 1
                
                # Case 1: y_i > y_j (True)
                if y_true[i] > y_true[j]:
                    if y_pred[i] > y_pred[j]: # Concordant
                        c_index += 1
                    elif y_pred[i] == y_pred[j]: # Tie in prediction (partial credit)
                        c_index += 0.5
                        
                # Case 2: y_i < y_j (True)
                elif y_true[i] < y_true[j]:
                    if y_pred[i] < y_pred[j]: # Concordant
                        c_index += 1
                    elif y_pred[i] == y_pred[j]: # Tie in prediction (partial credit)
                        c_index += 0.5
                        
    return c_index / n_pairs if n_pairs > 0 else 0.5

# --- 新增 R2m Score (R2m(test)) 计算函数 ---
def r2m_score(y_true, y_pred):
    """
    Calculates the R2m score (R2m(test) or R2m(adj)) using R2 and R0^2 (RTO R-squared).
    R2m = R2 * (1 - sqrt(|R2 - R0^2|))
    """
    r2 = r2_score(y_true, y_pred)
    
    # 1. Calculate R0^2 (R-squared through the origin)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 避免除以零
    if np.sum(y_pred**2) == 0:
        return 0.0
        
    # 计算 RTO 斜率 m
    m = np.sum(y_true * y_pred) / np.sum(y_pred**2)
    
    # RTO 残差平方和
    ss_res_rto = np.sum((y_true - m * y_pred)**2)
    # 总平方和
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    r0_2 = 1 - (ss_res_rto / ss_tot) if ss_tot != 0 else 0.0

    # 2. Calculate R2m
    # R2m = R2 * (1 - sqrt(|R2 - R0^2|))
    r2m = r2 * (1 - np.sqrt(np.abs(r2 - r0_2)))
    
    # 确保 R2m >= 0.0
    return r2m if r2m >= 0.0 else 0.0 


def calculate_regression_metrics(true_values, pred_values):
    """计算回归指标：RMSE, MAE, R2, R, C-index, R2m"""
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)

    # MSE and derived metrics
    mse = mean_squared_error(true_values, pred_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, pred_values)
    r2 = r2_score(true_values, pred_values)

    # Pearson Correlation Coefficient (R)
    if np.std(true_values) == 0 or np.std(pred_values) == 0:
        r_value = 0.0 # If one variable is constant
    else:
        # np.corrcoef returns the correlation matrix
        r_value = np.corrcoef(true_values, pred_values)[0, 1]
    
    # C-index
    ci_score = concordance_index(true_values, pred_values)
    
    # R2m Score
    r2m_score_val = r2m_score(true_values, pred_values)

    return {
        'RMSE': rmse, 
        'MAE': mae, 
        'R2': r2,
        'R': r_value, 
        'CI': ci_score,
        'R2m': r2m_score_val # 新增 R2m
    }

def get_mean_and_std(data_list):
    """计算列表数据的平均值和标准差。"""
    data = np.array(data_list)
    return np.mean(data), np.std(data)