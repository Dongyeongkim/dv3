import math
import torch
import torch.nn as nn


class Addition(nn.Module):
    def __init__(self, add_val):
        super().__init__()
        self.add_val = add_val
    def forward(self, x):
        return x + self.add_val

# Return proper activation functions 

# should fix this function to get nice and proper output

def conv_shape_calc(input_size: int, kernel_size: int, stride_size: int, padding_size: int):
    return math.floor((input_size - kernel_size + 2*padding_size)/stride_size) + 1

def deconv_shape_calc(input_size: int, kernel_size: int, stride_size: int, padding_size: int, output_padding_size: int):
    return stride_size*(input_size - 1) + kernel_size - 2*padding_size + output_padding_size


def return_func_from_name(name: str or int or float, ch: int, sh: int):

    if name == 'relu':
        return [nn.ReLU()]
    
    elif name == 'elu':
        return [nn.ELU()]
    
    elif name == 'sigmoid':
        return [nn.Sigmoid()]
    
    elif name == 'layernorm+silu':
        return [nn.LayerNorm([ch, sh, sh]), nn.SiLU()]
    
    elif name == 'layernorm(mlp)+silu':
        return [nn.LayerNorm([ch, sh]), nn.SiLU()]
    
    elif ((type(name) == int) or (type(name) == float)):
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
        n = torch.tanh(self.ln_n(ni+r*nh))
        return (1-z)*n + z*h


class Encoder(nn.Module):
    def __init__(self, input_size: int, in_channel: int, channels: list, kernels: list, 
                 strides: list, paddings: list, activations: list):
        super().__init__()
        enc = []
        shp = input_size
        enc.append(nn.Conv2d(in_channels=in_channel, out_channels=channels[0], 
                             kernel_size=kernels[0], stride=strides[0], padding=paddings[0]))
        shp = conv_shape_calc(shp, kernel_size=kernels[0], stride_size=strides[0], padding_size=paddings[0])
        enc.extend(return_func_from_name(activations[0], ch=channels[0], sh=shp)) 

        for i in range(len(channels)-1):
            enc.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                 kernel_size=kernels[i+1], stride=strides[i+1], padding=paddings[i+1]))
            shp = conv_shape_calc(shp, kernel_size=kernels[i+1], stride_size=strides[i+1], padding_size=paddings[i+1])
            enc.extend(return_func_from_name(activations[i+1], ch=channels[i+1], sh=shp))
        
        self.enc = nn.Sequential(*enc)
    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, input_size: int, out_channel: int, channels: list, kernels: list,
                  strides: list, paddings: list, activations: list):
        super().__init__()
        dec = []
        shp = input_size
        for i in range(len(channels)-1):
            dec.append(nn.ConvTranspose2d(in_channels=channels[i], out_channels=channels[i+1],
                                 kernel_size=kernels[i], stride=strides[i], padding=paddings[i], output_padding=paddings[i]))
            shp = deconv_shape_calc(shp, kernel_size=kernels[i], stride_size=strides[i], padding_size=paddings[i],
                                    output_padding_size=paddings[i])
            dec.extend(return_func_from_name(activations[i],ch=channels[i+1], sh=shp))
        
        dec.append(nn.ConvTranspose2d(in_channels=channels[-1], out_channels=out_channel,
                             kernel_size=kernels[-1], stride=strides[-1], padding=paddings[-1], output_padding=paddings[-1]))
        shp = deconv_shape_calc(shp, kernel_size=kernels[-1], stride_size=strides[-1], padding_size=paddings[-1],
                                 output_padding_size=paddings[-1])

        dec.extend(return_func_from_name(activations[-1], ch=channels[-1], sh=shp))

        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(x)


class DynamicsPredictor(nn.Module):
    def __init__(self, input_unit: int, num_unit: list, activations: list):
        super().__init__()
        dypre = []
        dypre.append(nn.Linear(input_unit, num_unit[0]))
        dypre.extend(return_func_from_name(activations[0], ch=1, sh=num_unit[0]))
        for i in range(len(num_unit)-1):
            dypre.append(nn.Linear(num_unit[i], num_unit[i+1]))
            dypre.extend(return_func_from_name(activations[i+1], ch=1, sh=num_unit[i+1]))    
         
        self.dypre = nn.Sequential(*dypre)
    
    def forward(self, x):
        return self.dypre(x)


class RewardPredictor(nn.Module):
    def __init__(self, input_unit: int, num_unit: list, activations: list):
        super().__init__()
        repre = []
        repre.append(nn.Linear(input_unit, num_unit[0]))
        repre.extend(return_func_from_name(activations[0], ch=1, sh=num_unit[0]))
        for i in range(len(num_unit)-1):
            repre.append(nn.Linear(num_unit[i], num_unit[i+1]))
            repre.extend(return_func_from_name(activations[i+1], ch=1, sh=num_unit[i+1]))
        
        self.repre = nn.Sequential(*repre)
    
    def forward(self, x):
        return self.repre(x)


class DiscountPredictor(nn.Module):
    def __init__(self, input_unit: int, num_unit: list, activations: list):
        super().__init__()
        dipre = []
        dipre.append(nn.Linar(input_unit, num_unit[0]))
        dipre.extend(return_func_from_name(activations[0], ch=1, sh=num_unit[0]))
        for i in range(len(num_unit)-1):
            dipre.append(nn.Linear(num_unit[i], num_unit[i+1]))
            dipre.extend(return_func_from_name(activations[i+1], ch=1, sh=num_unit[i+1]))
        
        self.dipre = nn.Sequential(*dipre)
    
    def forward(self, x):
        return self.dipre(x)
    


class ContinuousSequenceModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, latent_dim: int, use_layer_norm: bool, gru_recurrent_units: bool):
        super().__init__()

        
        self._mu = nn.Linear(input_shape, latent_dim)
        self._sig = nn.Linear(input_shape, latent_dim)
        self._out = nn.Linear(gru_recurrent_units, output_shape)

        if use_layer_norm:
            self.gruCell = nn.GRUCell(input_size=latent_dim, hidden_size=gru_recurrent_units)
        else:
            self.gruCell = GRU_ln_Cell(input_dim=latent_dim, hidden_dim=gru_recurrent_units)
        
    def forward(self, x, h):
        std = self._sig(x)
        z = self._mu(x) + std*torch.randn_like(std)
        h = self.gruCell(z, h)
        out = self._out(h)
        return out, h

class DiscreteSequenceModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, latent_dim: int, use_layer_norm: bool, gru_recurrent_units: bool):
        super().__init__()

        self.codebook = torch.eye(latent_dim)

        if use_layer_norm:
            self.gruCell = nn.GRUCell(input_size=latent_dim, hidden_size=gru_recurrent_units)
        else:
            self.gruCell = GRU_ln_Cell(input_dim=latent_dim, hidden_dim=gru_recurrent_units)
    
    def forward(self, x, h):
        pass


class SequenceModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, discrete: bool, latent_dim: int, use_layer_norm: bool, gru_recurrent_units: bool):
        super().__init__()
        
        if discrete:
            self.seq = DiscreteSequenceModel(input_shape, output_shape, latent_dim, use_layer_norm, gru_recurrent_units)
        else:
            self.seq = ContinuousSequenceModel(input_shape, output_shape, latent_dim, use_layer_norm, gru_recurrent_units)
    
    def forward(self, x, h):
        return self.seq(x, h)


            





if __name__ == "__main__":

    # Check every method is working well 

    print(return_func_from_name('elu', 32, 3))
    print(return_func_from_name('relu', 32, 3))
    print(return_func_from_name('sigmoid', 32, 3))
    print(return_func_from_name('layernorm+silu', 32, 3))
    print(return_func_from_name(0.5, 32, 3))
    try:
        print(return_func_from_name('relux', 32, 3))
    except NotImplementedError:
        print("Checked that showing NotImplementedError properly")

    # Check encoders

    from omegaconf import OmegaConf

    conf = OmegaConf.load('model.yaml')
    
    dv1_enc_conf = conf['model']['enc']['dv1']
    dv2_enc_conf = conf['model']['enc']['dv2']
    dv3_enc_conf = conf['model']['enc']['dv3']['m']

    dv1Enc = Encoder(*dv1_enc_conf.values())
    dv2Enc = Encoder(*dv2_enc_conf.values())
    dv3Enc = Encoder(*dv3_enc_conf.values())

    print(dv1Enc, dv2Enc, dv3Enc)
    
    # Check decoders 

    dv1_dec_conf = conf['model']['dec']['dv1']
    dv2_dec_conf = conf['model']['dec']['dv2']
    dv3_dec_conf = conf['model']['dec']['dv3']['m']

    dv1Dec = Decoder(*dv1_dec_conf.values())
    dv2Dec = Decoder(*dv2_dec_conf.values())
    dv3Dec = Decoder(*dv3_dec_conf.values())

    print(dv1Dec, dv2Dec, dv3Dec)

    test_enc_input = torch.randn(32, 1, 64, 64)
    dv1_er = dv1Enc(test_enc_input)
    dv2_er = dv2Enc(test_enc_input)
    dv3_er = dv3Enc(test_enc_input)
    
    print(dv1_er.shape, dv2_er.shape, dv3_er.shape)
    
    test_dec_input_dv1 = torch.randn(32, 1024, 1, 1)
    test_dec_input_dv2 = torch.randn(32, 1024, 1, 1)
    test_dec_input_dv3 = torch.randn(32, 384,  4, 4)

    dv1_re = dv1Dec(test_dec_input_dv1)
    dv2_re = dv2Dec(test_dec_input_dv2)
    dv3_re = dv3Dec(test_dec_input_dv3)

    print(dv1_re.shape, dv2_re.shape, dv3_re.shape)

    dv1_dyn_conf = conf['model']['dyn']['dv1']
    dv2_dyn_conf = conf['model']['dyn']['dv2']
    dv3_dyn_conf = conf['model']['dyn']['dv3']['m']

    dv1Dyn = DynamicsPredictor(200, *dv1_dyn_conf.values())
    dv2Dyn = DynamicsPredictor(600, *dv2_dyn_conf.values())
    dv3Dyn = DynamicsPredictor(1024, *dv3_dyn_conf.values())

    print(dv1Dyn, dv2Dyn, dv3Dyn)

    test_dyn_input_dv1 = torch.randn(32, 1, 200)
    test_dyn_input_dv2 = torch.randn(32, 1, 600)
    test_dyn_input_dv3 = torch.randn(32, 1, 1024)

    dv1_dyn_out = dv1Dyn(test_dyn_input_dv1)
    dv2_dyn_out = dv2Dyn(test_dyn_input_dv2)
    dv3_dyn_out = dv3Dyn(test_dyn_input_dv3)

    print(dv1_dyn_out.shape, dv2_dyn_out.shape, dv3_dyn_out.shape)
    