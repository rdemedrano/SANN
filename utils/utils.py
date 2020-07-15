import os
import json
import torch
import numpy as np
import shutil
from collections import defaultdict


def model_summary(model):
    """
    Model layers and parameters information
    """
    print("Model summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    print("="*100)
    print(f"Total Params:{total_param}")       
          
class DotDict(dict):
    """
    Dot notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
class Logger(object):
    """
    Log information through the process
    """
    def __init__(self, log_dir, chkpt_interval):
        super(Logger, self).__init__()
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(os.path.join(log_dir))
        self.log_path = os.path.join(log_dir, 'logs.json')
        self.model_path = os.path.join(log_dir, 'model.pth')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0
        self.chkpt_interval = chkpt_interval

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] += 1

    def save(self, model):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        torch.save(model.state_dict(), self.model_path)