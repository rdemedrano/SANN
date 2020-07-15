"""
#######################################################################################################################
# METRICS: Expect a NxTxS tensor
#######################################################################################################################
"""

def mae(x_pred, x_target, dim=0):
    """
    MAE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).abs().mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).abs().mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).abs().mean((0,2))
    else:
        raise ValueError("Not a valid dimension")

def wmape(x_pred, x_target, dim=0):
    """
    WMAPE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return 100*(mae(x_pred, x_target, dim = dim)/(x_target.abs().mean())).item()
    elif dim == 1:
        return 100*(mae(x_pred, x_target, dim = 1)/(x_target.abs().mean((0,1))))
    elif dim == 2:
        return 100*(mae(x_pred, x_target)/(x_target.abs().mean((0,2))))
    else:
        raise ValueError("Not a valid dimension")


def rmse(x_pred, x_target, dim=0):
    """
    RMSE calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).pow(2).mean().sqrt().item()
    elif dim == 1:
        return x_pred.sub(x_target).pow(2).mean((0,1)).sqrt().squeeze()
    elif dim == 2:
        return x_pred.sub(x_target).pow(2).mean((0,2)).sqrt().squeeze()
    else:
        raise ValueError("Not a valid dimension")

def bias(x_pred, x_target, dim=0):
    """
    Bias calculation
    If dim == 0, overall MAE. 
    If dim == 1, by spatial point.
    If dim == 2, by timestep.
    """
    if dim == 0:
        return x_pred.sub(x_target).mean().item()
    elif dim == 1:
        return x_pred.sub(x_target).mean((0,1))
    elif dim == 2:
        return x_pred.sub(x_target).mean((0,2))
    else:
        raise ValueError("Not a valid dimension")