import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset):
    train_file = os.path.join('./data/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('./data/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

def load_HAR():
    
    def extract_data(path):
        res_data = []
        # res_labels = []

        x_path = path + 'InertialSignals/'
        for f in os.listdir(x_path):
            dati = []
            with open(x_path + f, 'r') as fp:
                for line in fp.readlines():
                    # print(line)
                    dati.append([float(i) for i in line.split(' ') if i != ''])
            
            res_data.append(dati)
        
        with open(path + 'y.txt', 'r') as fp:
            res_labels = [float(i) for i in fp.readlines() if i != '']

        return np.array(res_data).swapaxes(1, 0), np.array(res_labels)
    
    train_X, train_y = extract_data(f'./data/HAR/har/train/')
    test_X, test_y = extract_data(f'./data/HAR/har/test/')
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_UEA(dataset):
    # UEA数据集使用.ts文件格式，需要特殊解析
    train_file = f'./data/UEA/{dataset}/{dataset}_TRAIN.ts'
    test_file = f'./data/UEA/{dataset}/{dataset}_TEST.ts'
    
    def parse_ts_file(file_path):
        """解析.ts文件格式"""
        data_lines = []
        labels = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过元数据行（以@开头）
                if line.startswith('@'):
                    continue
                
                # 解析数据行：特征:标签
                if ':' in line:
                    # 使用rsplit找到最后一个冒号，分割特征和标签
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        features_str, label = parts
                        # 处理多重冒号的情况 - 将所有特征部分合并
                        all_features = []
                        for part in features_str.split(':'):
                            if part.strip():
                                try:
                                    features = [float(x) for x in part.split(',') if x.strip()]
                                    all_features.extend(features)
                                except ValueError:
                                    continue
                        
                        if all_features:
                            data_lines.append(all_features)
                            labels.append(label)
        
        return np.array(data_lines), np.array(labels)
    
    # 解析训练和测试数据
    train_X, train_y = parse_ts_file(train_file)
    test_X, test_y = parse_ts_file(test_file)
    
    # 将标签字符串转换为整数
    unique_labels = np.unique(np.concatenate([train_y, test_y]))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    train_y = np.array([label_to_int[label] for label in train_y])
    test_y = np.array([label_to_int[label] for label in test_y])
    
    # 转换为正确的数据类型
    train_X = train_X.astype(np.float32)
    train_y = train_y.astype(np.int32)
    test_X = test_X.astype(np.float32)
    test_y = test_y.astype(np.int32)
    
    # 标准化
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    # 重塑为时间序列格式 (samples, timesteps, features)
    # 这里假设数据是单变量的，需要添加通道维度
    train_X = train_X.reshape(-1, train_X.shape[1], 1)
    test_X = test_X.reshape(-1, test_X.shape[1], 1)
    
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'./data/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(np.float32)


def load_forecast_csv(name, univar=False):
    # ETT数据集文件位于 ./data/ETT/ 目录下
    if name.startswith('ETT'):
        data = pd.read_csv(f'./data/ETT/{name}.csv', index_col='date', parse_dates=True)
    else:
        data = pd.read_csv(f'./data/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'./data/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
