from models.sann import SANN


def define_model(opt, *args):
    if opt.name == 'SANN':
        return SANN(opt.n_inp, opt.n_out, opt.t_inp, opt.t_out, opt.n_points, opt.past_t, opt.hidden_dim, opt.drop)
    else:
        raise ValueError("Not a known model")

