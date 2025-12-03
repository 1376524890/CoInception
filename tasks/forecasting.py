import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    padding = 200
    
    t = time.time()
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    coinception_infer_time = time.time() - t
    
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        original_shape = test_pred.shape
        
        # 对于多变量且每个变量单独处理的情况（如electricity数据集）
        if test_data.shape[0] > 1 and original_shape[-1] == 1:
            # 重塑为 (变量数, 时间步数*预测长度, 1)
            test_pred_3d = test_pred.reshape(test_data.shape[0], -1, 1)
            test_labels_3d = test_labels.reshape(test_data.shape[0], -1, 1)
            
            # 转置为 (时间步数*预测长度, 变量数) 以匹配scaler的期望形状
            test_pred_2d = test_pred_3d.squeeze(-1).T
            test_labels_2d = test_labels_3d.squeeze(-1).T
            
            # 应用逆变换
            test_pred_inv_2d = scaler.inverse_transform(test_pred_2d)
            test_labels_inv_2d = scaler.inverse_transform(test_labels_2d)
            
            # 转回原始形状
            test_pred_inv_3d = test_pred_inv_2d.T.reshape(test_data.shape[0], -1, 1)
            test_labels_inv_3d = test_labels_inv_2d.T.reshape(test_data.shape[0], -1, 1)
            
            test_pred_inv = test_pred_inv_3d.reshape(original_shape)
            test_labels_inv = test_labels_inv_3d.reshape(original_shape)
        else:
            # 正常情况：重塑为二维数组以适应StandardScaler
            test_pred_2d = test_pred.reshape(-1, original_shape[-1])
            test_labels_2d = test_labels.reshape(-1, original_shape[-1])
            # 应用逆变换
            test_pred_inv_2d = scaler.inverse_transform(test_pred_2d)
            test_labels_inv_2d = scaler.inverse_transform(test_labels_2d)
            # 恢复原始形状
            test_pred_inv = test_pred_inv_2d.reshape(original_shape)
            test_labels_inv = test_labels_inv_2d.reshape(original_shape)
            
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'coinception_infer_time': coinception_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
