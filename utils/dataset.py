import torch
import numpy as np
from sklearn.model_selection import KFold
import itertools

from .scalers import MinMaxScaler, ZScaler, LogScaler


def k_cross(data, t_inp, t_out, n_points, k, n_inp, norm, sp_norm):
    """
    Data preparation for the k-cross strategy with a train/val/test split as validation scheme.
    """
    if data.dim() == 2:
        data.unsqueeze_(0)
    C,T,S = data.size()
    
    kf = KFold(n_splits=10, shuffle=False)
    
    n_data = T - 1 - (t_inp+t_out)

    train_indexes, test_indexes = next(itertools.islice(kf.split(data[:,0:n_data][0]), k, None))
    if k == 0:
        train_indexes = train_indexes[(t_inp + t_out):]
    elif k == (kf.n_splits-1):
        train_indexes = train_indexes[:-(t_inp + t_out)]
    else:
        middle_index = test_indexes[0]
        rem_index = np.arange(middle_index-(t_inp+t_out), middle_index+(t_inp+t_out))
        train_indexes = np.delete(train_indexes, rem_index)
        
    train_indexes, val_indexes = train_indexes[:int(len(train_indexes)*0.9)], train_indexes[int(len(train_indexes)*0.9):]
    
    train = []
    val = []
    test = []
    for train_index in train_indexes:
        train.append(data.unsqueeze(0)[:,:,train_index:train_index+(t_inp+t_out)]) 
    for val_index in val_indexes: 
        val.append(data.unsqueeze(0)[:,:,val_index:val_index+(t_inp+t_out)])
    for test_index in test_indexes: 
        test.append(data.unsqueeze(0)[:,:,test_index:test_index+(t_inp+t_out)])
    
    train = torch.cat(train, dim=0)
    val = torch.cat(val, dim=0)
    test = torch.cat(test, dim=0)
    
    
    if norm == 'min-max':
        sc = MinMaxScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'z':
        sc = ZScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'log':
        sc = LogScaler()
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
        
  
    X_train, Y_train = torch.split(data_train, t_inp, dim = 2)
    X_val, Y_val = torch.split(data_val, t_inp, dim = 2)
    X_test, Y_test = torch.split(data_test, t_inp, dim = 2)
    
    
    return (X_train.float(), Y_train.float(), X_val.float(), Y_val.float(), X_test.float(), Y_test.float(), sc)


def rolling_origin(data, t_inp, t_out, n_points, k, n_inp, norm, sp_norm):
    """
    Data preparation for the rolling origin strategy with a train/val/test split as validation scheme.
    For k=0 a minimun of 50% of the data is used.
    """
    if data.dim() == 2:
        data.unsqueeze_(0)
    C,T,S = data.size()
        

    n_data = T - 1 - (t_inp+t_out)
    last_index = int(n_data*(0.5+(k+1)*(1/20)))
    last_index_train = int(last_index*0.7)
    last_index_val = int(last_index*0.85)
        
    train_indexes = np.arange(0,last_index_train)
    val_indexes = np.arange(last_index_train+1, last_index_val)
    test_indexes = np.arange(last_index_val+1, last_index) 
        
    train = []
    val = []
    test = []
    for train_index in train_indexes:
        train.append(data.unsqueeze(0)[:,:,train_index:train_index+(t_inp+t_out)]) 
    for val_index in val_indexes: 
        val.append(data.unsqueeze(0)[:,:,val_index:val_index+(t_inp+t_out)])
    for test_index in test_indexes: 
        test.append(data.unsqueeze(0)[:,:,test_index:test_index+(t_inp+t_out)])
    
    train = torch.cat(train, dim=0)
    val = torch.cat(val, dim=0)
    test = torch.cat(test, dim=0)
    
    
    if norm == 'min-max':
        sc = MinMaxScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'z':
        sc = ZScaler(by_point = sp_norm)
        sc.fit(train)
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
    elif norm == 'log':
        sc = LogScaler()
        data_train = sc.scale(train)
        data_val = sc.scale(val)
        data_test = sc.scale(test)
  
    X_train, Y_train = torch.split(data_train, t_inp, dim = 2)
    X_val, Y_val = torch.split(data_val, t_inp, dim = 2)
    X_test, Y_test = torch.split(data_test, t_inp, dim = 2)
    
    return (X_train.float(), Y_train.float(), X_val.float(), Y_val.float(), X_test.float(), Y_test.float(), sc)