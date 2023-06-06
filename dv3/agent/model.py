import torch
import torch.nn as nn
import torch.nn.functional as F

# Return proper activation functions 

class Addition(nn.Module):
    def __init__(self, add_val):
        super().__init__()
        self.add_val = add_val
    def forward(self, x):
        return x + self.add_val

def return_func_from_name(name: str or int, ch: int, it: int):

    if name == 'relu':
        return [nn.ReLU()]
    
    elif name == 'sigmoid':
        return [nn.Sigmoid()]
    
    elif name == 'layernorm+silu':
        return [nn.LayerNorm([ch, 2**(5-it), 2**(5-it)]), nn.SiLU()]
    
    elif name == 0.5:
        return [Addition(name)]
    
    else:
        raise NotImplementedError

class GRU_ln_Cell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.ih = nn.Linear(input_dim, 3*hidden_dim)
        self.hh = nn.Linear(hidden_dim, 3*hidden_dim)
        self.ln_r = nn.LayerNorm(hidden_dim)
        self.ln_z = nn.LayerNorm(hidden_dim)
        self.ln_n = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, h):
        ri, zi, ni = torch.chunk(self.ih(x), 3, dim=1)
        rh, zh, nh = torch.chunk(self.hh(h), 3, dim=1)
        r = torch.sigmoid(self.ln_r(ri+rh))
        z = torch.sigmoid(self.ln_z(zi+zh))
        n = torch.tanh(ni+r*nh)
        return (1-z)*n + z*h


class Encoder(nn.Module):
    def __init__(self, in_channel: int, channels: list, kernels: list, strides: list, paddings: list, activations: list):
        super().__init__()
        enc = []
        enc.append(nn.Conv2d(in_channels=in_channel, out_channels=channels[0], 
                             kernel_size=kernels[0], stride=strides[0], padding=paddings[0]))
        enc.extend(return_func_from_name(activations[0], ch=channels[0], it=0)) 

        for i in range(len(channels)-1):
            enc.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                 kernel_size=kernels[i+1], stride=strides[i+1], padding=paddings[i+1]))
            enc.extend(return_func_from_name(activations[i+1], ch=channels[i+1], it=i+1))
        
        self.enc = nn.Sequential(*enc)

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, out_channel: int, channels: list, kernels: list, strides: list, paddings: list, activations: list):
        super().__init__()
        dec = []
        for i in range(len(channels)-1):
            dec.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                 kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
            dec.extend(return_func_from_name(activations[i],ch=channels[i], it=i))
        
        dec.append(nn.Conv2d(in_channels=channels[-1], out_channels=out_channel,
                             kernel_size=kernels[-1], stride=strides[-1], padding=paddings[-1]))
        dec.extend(return_func_from_name(activations[-1], ch=channels[-1], it=-1))

        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(x)


class DynamicsPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass


class RewardPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

class ContinuePredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass



class RSSM(nn.Module):
    def __init__(self, inputs: int, hiddens: int, use_layer_norm: bool):
        super().__init__()
        if use_layer_norm:
            self.gruCell = GRU_ln_Cell(input_dim=inputs, hidden_dim=hiddens)
        else:
            self.gruCell = nn.GRUCell(input_size=inputs, hidden_size=hiddens) 

    def forward(self, x):
        pass





if __name__ == "__main__":

    # Check every method is working well 

    print(return_func_from_name('relu', 32, 3))
    print(return_func_from_name('sigmoid', 32, 3))
    print(return_func_from_name('layernorm+silu', 32, 3))
    print(return_func_from_name(0.5, 32, 3))
    # print(return_func_from_name('relux', 32, 3))

    # Check encoders

    from omegaconf import OmegaConf

    conf = OmegaConf.load('model.yaml')
    
    dv1_enc_conf = conf['model']['enc']['dv1']
    dv2_enc_conf = conf['model']['enc']['dv2']
    dv3_enc_conf = conf['model']['enc']['dv3']['m']

    dv1Enc = Encoder(1, *dv1_enc_conf.values())
    dv2Enc = Encoder(1, *dv2_enc_conf.values())
    dv3Enc = Encoder(1, *dv3_enc_conf.values())

    print(dv1Enc, dv2Enc, dv3Enc)
    # Check decoders 

    dv1_dec_conf = conf['model']['dec']['dv1']
    dv2_dec_conf = conf['model']['dec']['dv2']
    dv3_dec_conf = conf['model']['dec']['dv3']['m']

    dv1Dec = Decoder(1, *dv1_dec_conf.values())
    dv2Dec = Decoder(1, *dv2_dec_conf.values())
    dv3Dec = Decoder(1, *dv3_dec_conf.values())

    print(dv1Dec, dv2Dec, dv3Dec)