import torch


"""
#######################################################################################################################
# MIN-MAX SCALER
#######################################################################################################################
"""
class MinMaxScaler():
    """
    Min-Max scaler for tensors with CxTxS or NxCxTxS format.
    N: batch
    C: number of input series
    T: input lags
    S: number of spatial locations
    """
    def __init__(self, by_point = True, new_range = (0,1)):
        self.min = new_range[0]
        self.max = new_range[1]
        self.by_point = by_point
        
        self.min_data = None
        self.max_data = None
        
        self.is_fitted = False
        self.is_scaled = False
        
    def fit(self, data):
        self.is_fitted = True        
        if data.dim() == 3:
            # C,T,S format
            self.min_data = torch.min(data, dim = 1).values
            self.max_data = torch.max(data, dim = 1).values
            if not self.by_point:
                self.min_data = torch.min(self.min_data, dim = 1).values
                self.max_data = torch.max(self.max_data, dim = 1).values
        elif data.dim() == 4:
            # N,C,T,S format
            self.min_data = torch.min(data, dim = 2).values
            self.max_data = torch.max(data, dim = 2).values
            self.min_data = torch.min(self.min_data, dim = 0).values
            self.max_data = torch.max(self.max_data, dim = 0).values
            if not self.by_point:
                self.min_data = torch.min(self.min_data, dim = 1).values
                self.max_data = torch.max(self.max_data, dim = 1).values
        else:
            self.is_fitted = False
            raise ValueError("Not a spatio-temporal sequence known")
            
    def scale(self, data):
        if self.is_fitted:
            self.is_scaled = True
            if not self.by_point:
                if data.dim() == 3:
                    # C,T,S format
                    C,T,S = data.size()
                    return (((data.view(C,-1) - self.min_data.unsqueeze(1))/(self.max_data.unsqueeze(1) - self.min_data.unsqueeze(1))) * (self.max - self.min) + self.min).view(C,T,S)
                elif data.dim() == 4:
                    # N,C,T,S format
                    N,C,T,S = data.size()
                    min_rep = self.min_data.view(1,C,1,1).repeat(N,1,1,1)
                    max_rep = self.max_data.view(1,C,1,1).repeat(N,1,1,1)
                    return (((data - min_rep)/(max_rep - min_rep)) * (self.max - self.min) + self.min)
                else:
                    raise ValueError("Not a spatio-temporal sequence known")
            if data.dim() == 3:
                # C,T,S format
                return ((data - self.min_data.unsqueeze(1))/(self.max_data.unsqueeze(1) - self.min_data.unsqueeze(1))) * (self.max - self.min) + self.min
            elif data.dim() == 4:
                # N,C,T,S format
                N,C,T,S = data.size()
                return ((data - self.min_data.view(1,C,1,S))/(self.max_data.view(1,C,1,S) - self.min_data.view(1,C,1,S))) * (self.max - self.min) + self.min
            else:
                raise ValueError("Not a spatio-temporal sequence known")
        else:
            raise ValueError("You need to fit your scaler first")
            
    def rev_scale(self, data):
        if self.is_fitted:
            if self.is_scaled:
                if not self.by_point:
                    if data.dim() == 3:
                        # C,T,S format
                        C,T,S = data.size()
                        return (self.min_data.unsqueeze(1) + ((data.view(C,-1) - self.min).mul(self.max_data.unsqueeze(1) - self.min_data.unsqueeze(1)))/(self.max - self.min)).view(C,T,S)
                    elif data.dim() == 4:
                        # N,C,T,S format
                        N,C,T,S = data.size()
                        min_rep = self.min_data[:C].view(1,C,1,1).repeat(N,1,1,1)
                        max_rep = self.max_data[:C].view(1,C,1,1).repeat(N,1,1,1)
                        return min_rep + ((data - self.min).mul(max_rep - min_rep))/(self.max - self.min)
                    else:
                        raise ValueError("Not a spatio-temporal sequence known")
                
                if data.dim() == 3:
                    # C,T,S format
                    return self.min_data.unsqueeze(1) + ((data - self.min).mul(self.max_data.unsqueeze(1) - self.min_data.unsqueeze(1)))/(self.max - self.min)
                elif data.dim() == 4:
                    # N,C,T,S format
                    N,C,T,S = data.size()
                    return self.min_data[:C].view(1,C,1,S) + ((data - self.min).mul(self.max_data[:C].view(1,C,1,S) - self.min_data[:C].view(1,C,1,S)))/(self.max - self.min)
                else:
                    raise ValueError("Not a spatio-temporal sequence known")
            else:
                raise ValueError("You need to scale your data first!")
        else:
            raise ValueError("You need to fit your scaler first!")
            
            
            
"""
#######################################################################################################################
# Z SCALER
#######################################################################################################################
"""   
class ZScaler():
    """
    Z scaler for tensors with CxTxS or NxCxTxS format.
    N: batch
    C: number of input series
    T: input lags
    S: number of spatial locations
    """
    def __init__(self, by_point = True):
        self.by_point = by_point
        
        self.std = None
        self.mean = None
        
        self.is_fitted = False
        self.is_scaled = False
        
    def fit(self, data):
        self.is_fitted = True
        if data.dim() == 3:
            # C,T,S format
            if self.by_point:
                self.std = torch.std(data, dim = 1)
                self.mean = torch.mean(data, dim = 1)
            else:
                self.std = torch.std(data, dim = (1,2))
                self.mean = torch.mean(data, dim = (1,2))
        elif data.dim() == 4:
            # N,C,T,S format
            if self.by_point:
                self.std = torch.std(data, dim = (0,2))
                self.mean = torch.mean(data, dim = (0,2))
            else:
                self.std = torch.std(data, dim = (0,2,3))
                self.mean = torch.mean(data, dim = (0,2,3))
        else:
            self.is_fitted = False
            raise ValueError("Not a spatio-temporal sequence known")
            
                
    def scale(self, data):
        if self.is_fitted:
            self.is_scaled = True
            if not self.by_point:
                if data.dim() == 3:
                    # C,T,S format
                    C,T,S = data.size()
                    return ((data.view(C,-1) - self.mean.unsqueeze(1))/self.std.unsqueeze(1)).view(C,T,S)
                elif data.dim() == 4:
                    # N,C,T,S format
                    N,C,T,S = data.size()
                    mean_rep = self.mean.view(1,C,1,1).repeat(N,1,1,1)
                    std_rep = self.std.view(1,C,1,1).repeat(N,1,1,1)
                    return (data - mean_rep)/std_rep
                else:
                    raise ValueError("Not a spatio-temporal sequence known")
            if data.dim() == 3:
                # C,T,S format
                return (data - self.mean.unsqueeze(1))/self.std.unsqueeze(1)
            elif data.dim() == 4:
                # N,C,T,S format
                N,C,T,S = data.size()
                return (data - self.mean.view(1,C,1,S))/self.std.view(1,C,1,S)
            else:
                    raise ValueError("Not a spatio-temporal sequence known")
        else:
            raise ValueError("You need to fit your scaler first!")
            
    def rev_scale(self, data):
        if self.is_fitted:
            if self.is_scaled:
                if not self.by_point:
                    if data.dim() == 3:
                        # C,T,S format
                        C,T,S = data.size()
                        return (data.view(C,-1).mul(self.std.unsqueeze(1)) + self.mean.unsqueeze(1)).view(C,T,S)
                    elif data.dim() == 4:
                        # N,C,T,S format
                        N,C,T,S = data.size()
                        mean_rep = self.mean[:C].view(1,C,1,1).repeat(N,1,1,1)
                        std_rep = self.std[:C].view(1,C,1,1).repeat(N,1,1,1)
                        return data.mul(std_rep) + mean_rep
                    else:
                        raise ValueError("Not a spatio-temporal sequence known")
                if data.dim() == 3:
                    # C,T,S format
                    return data.mul(self.std.unsqueeze(1)) + self.mean.unsqueeze(1)
                elif data.dim() == 4:
                    # N,C,T,S format
                    N,C,T,S = data.size()
                    return data.mul(self.std[:C].view(1,C,1,S)) + self.mean[:C].view(1,C,1,S)
                else:
                    raise ValueError("Not a spatio-temporal sequence known")
            else:
                raise ValueError("You need to scale your data first!")
        else:
            raise ValueError("You need to fit your scaler first!")
            
            
    
            
"""
#######################################################################################################################
# LOG SCALER
#######################################################################################################################
"""    
class LogScaler():
    """
    Log scaler for tensors
    """
    def __init__(self, eps = 0.999):
        self.eps = eps
        self.is_scaled = False

    def scale(self, data):
        self.is_scaled = True
        return torch.log(data + self.eps)
    
    def rev_scale(self, data):
        if self.is_scaled:
            return torch.exp(data) - self.eps
        else:
            raise ValueError("You need to scale your data first!")
