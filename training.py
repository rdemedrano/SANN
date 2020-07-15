import torch
import torch.optim as optim

import os
import json
import numpy as np
import configargparse
from collections import defaultdict
from tqdm import trange
import pickle
import datetime

from utils.dataset import k_cross, rolling_origin
from utils.utils import Logger, DotDict
from utils.metrics import rmse, bias, wmape
from utils.model import define_model

"""
#######################################################################################################################
# VARIABLES AND OPTIONS
#######################################################################################################################
"""
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='acpol')
p.add('--extension', type=str, help='dataset extension', default='.csv', choices=['.csv'])
# -- exp
p.add('--outputdir', type=str, help='path to save exp', default='output')
# -- model
p.add('--name', type=str, help='name of the model', default='SANN')
p.add('--second_name', type=str, help='second name of the model', default='')
p.add('--t_inp', type=int, help='number of input timesteps', default=7)
p.add('--t_out', type=int, help='number of output timesteps', default=1)
p.add('--n_points', type=int, help='number of spatial points/sensors', default=30)
p.add('--n_inp', type=int, help='number of input features', default=1)
p.add('--n_out', type=int, help='number of output features', default=1)
p.add('--past_t', type=int, help='number of lags', default=1)
p.add('--hidden_dim', type=list, help='hidden dimension', default=40)
p.add('--drop', type=float, help='2D dropout', default=0.0)
# -- evaluation
p.add('--rolling_origin', type=bool, help='wether to use rolling origin or not', default=False)
p.add('--k', type=int, help='fold of k-cross-validation', default=1)
p.add('--norm', type=str, help='type of normalization', default='z', choices=['min-max', 'z', 'log'])
p.add('--sp_norm', type=bool, help='wether normalization is by spatial point or not', default=True)
# -- optim
p.add('--lr', type=float, help='learning rate', default=1e-3)
p.add('--lr_t', type=float, help='learning rate threshold for decay', default=1e-5)
p.add('--beta1', type=float, help='adam beta1', default=.9)
p.add('--beta2', type=float, help='adam beta2', default=.999)
p.add('--eps', type=float, help='optim eps', default=1e-8)
p.add('--wd', type=float, help='weight decay', default=1e-3)
# -- learning
p.add('--batch_size', type=int, default=256, help='batch size')
p.add('--patience', type=int, default=10, help='number of epoch to wait before trigerring lr decay')
p.add('--n_epochs', type=int, default=100, help='number of epochs to train for')
# -- gpu
p.add('--device', type=int, default=0, help='-1: cpu; > -1: cuda device id')
# parse
opt = DotDict(vars(p.parse_args()))

if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

choices=['min-max', 'z', 'log']
if opt.norm not in choices:
    raise ValueError("Not a valid normalization technique")

opt.outputdir = opt.outputdir + '/' +  opt.name + opt.second_name + '/' + str(opt.k)


"""
#######################################################################################################################
# DATA PREPARATION
#######################################################################################################################
"""
if opt.extension == '.csv':
    data = torch.Tensor(np.genfromtxt(os.path.join(opt.datadir, opt.dataset + opt.extension)))
else:
    raise ValueError("Not a valid file extension for data")

allowed_dim = [2,3,4]
if data.dim() not in allowed_dim:
    raise ValueError("Unknown spatio-temporal data format")


if not opt.rolling_origin:
    X_train, Y_train, X_val, Y_val, X_test, Y_test, sc = k_cross(data, opt.t_inp, 
                                                                          opt.t_out, opt.n_points,
                                                                          opt.k, opt.n_inp,
                                                                          opt.norm, opt.sp_norm)
else:
    X_train, Y_train, X_val, Y_val, X_test, Y_test, sc = rolling_origin(data, opt.t_inp, 
                                                                          opt.t_out, opt.n_points,
                                                                          opt.k, opt.n_inp,
                                                                          opt.norm, opt.sp_norm)

train_dataset = []
for i in range(len(X_train)):
    train_dataset.append([X_train[i], Y_train[i]])
    
val_dataset = []
for i in range(len(X_val)):
    val_dataset.append([X_val[i], Y_val[i]]) 
    
test_dataset = []
for i in range(len(X_test)):
   test_dataset.append([X_test[i], Y_test[i]]) 
    
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = opt.batch_size,
                                           shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = len(X_val),
                                           shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = len(X_test),
                                           shuffle = False)


"""
#######################################################################################################################
# MODEL
#######################################################################################################################
"""
model = define_model(opt)
model = model.to(device)    


"""
#######################################################################################################################
# OPTIMIZER
#######################################################################################################################
"""
loss_fn = torch.nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(),  lr=opt.lr, eps=opt.eps, weight_decay=opt.wd, momentum=0.9)

if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


"""
#######################################################################################################################
# LOGGER
#######################################################################################################################
"""
logger = Logger(opt.outputdir, 25)
with open(os.path.join(opt.outputdir, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)

logger.log('name', opt.name + opt.second_name)
logger.log('k', opt.k)


with open(opt.outputdir + '/' + 'scaler.pkl', 'wb') as f:
    pickle.dump(sc, f, pickle.HIGHEST_PROTOCOL)

"""
#######################################################################################################################
# TRAINING AND VALIDATION
#######################################################################################################################
"""
lr = opt.lr
tr = trange(opt.n_epochs, position=0, leave=True)

for t in tr: 
    model.train()
    logs_train = defaultdict(float)
    for i, (x,y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        logs_train['mse'] += loss.item()
        
    # Logs training
    logs_train['mse'] /= (i+1)
    logger.log('train', logs_train)

    model.eval()
    logs_val = defaultdict(float)
    with torch.no_grad():
        for x,y in val_loader:
            if torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)
            else:
                x = x
                y = y
            y_pred = model(x)
            loss_val = loss_fn(y_pred, y)
            logs_val['mse'] = loss_val.item()
            
            # Logs evaluation
            logger.log('val', logs_val)

    # General information
    tr.set_postfix(train_mse = logs_train['mse'], val_mse=logs_val['mse'], 
                   train_rmse = np.sqrt(logs_train['mse']), val_rmse=np.sqrt(logs_val['mse']),
                   lr = lr)
    logger.checkpoint(model)
    
    if opt.patience > 0:
        lr_scheduler.step(logs_val['mse'])
        lr = optimizer.param_groups[0]['lr']
    if lr <= opt.lr_t:
        break
    
logger.log('train.coc', logs_train['mse'])
logger.log('run_time', str(datetime.timedelta(seconds=round(tr.format_dict['elapsed']))))   

"""
#######################################################################################################################
# TEST
#######################################################################################################################
"""

model.eval()
logs_test = defaultdict(float)
with torch.no_grad():        
    for x,y in test_loader:
        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)
        else:
            x = x
            y = y
        y_pred = model(x)
            
        y_pred_dnorm = sc.rev_scale(y_pred.cpu()).view(-1, opt.t_out, opt.n_points)
        y_dnorm =sc.rev_scale(y.cpu()).view(-1, opt.t_out, opt.n_points)

        loss_test = loss_fn(y_pred_dnorm, y_dnorm)
        
        logs_test['mse'] = loss_test.item()
        logs_test['rmse'] = rmse(y_pred_dnorm, y_dnorm)
        logs_test['bias'] = bias(y_pred_dnorm, y_dnorm)
        logs_test['wmape'] = wmape(y_pred_dnorm, y_dnorm)
        
        logger.log('test', logs_test)
        
print("\n\n================================================")
print(" *  Test MSE: ", logs_test['mse'],
      "\n *  Test RMSE: ", logs_test['rmse'],
      "\n *  Test Bias: ", logs_test['bias'],
      "\n *  Test WMAPE (%): ", logs_test['wmape'])
print("================================================\n")
print("\nNormalization:" , opt.norm, ', k:', opt.k)

logger.log('test.coc', loss_fn(y_pred,y).item())
logger.save(model)