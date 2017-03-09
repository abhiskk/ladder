import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Encoder(torch.nn.Module):
    def __init__(self, d_in, d_out, activation_type,
                 train_bn_scaling, noise_level):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.noise_level = noise_level

        # Encoder
        # Encoder only uses W matrix, no bias
        self.linear = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) / np.sqrt(d_in)

        # Batch Normalization
        # For Relu Beta of batch-norm is redundant, hence only Gamma is trained
        # For Softmax Beta, Gamma are trained
        # batch-normalization bias
        self.bn_normalize_clean = torch.nn.BatchNorm1d(d_out, affine=False)
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)
        self.bn_beta = Parameter(torch.FloatTensor(1, d_out))
        self.bn_beta.data.zero_()
        if self.train_bn_scaling:
            # batch-normalization scaling
            self.bn_gamma = Parameter(torch.FloatTensor(1, d_out))
            self.bn_gamma.data = torch.ones(self.bn_gamma.size())

        # Activation
        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError("invalid Acitvation type")

        # buffer for z_pre, z which will be used in decoder cost
        self.buffer_z_pre = None
        self.buffer_z = None
        # buffer for tilde_z which will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def bn_gamma_beta(self, x):
        ones = Parameter(torch.ones(x.size()[0], 1))
        t = x + ones.mm(self.bn_beta)
        if self.train_bn_scaling:
            t = torch.mul(t, ones.mm(self.bn_gamma))
        return t

    def forward_clean(self, h):
        z_pre = self.linear(h)
        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z_pre = z_pre.detach().clone()
        z = self.bn_normalize_clean(z_pre)
        self.buffer_z = z.detach().clone()
        z_gb = self.bn_gamma_beta(z)
        h = self.activation(z_gb)
        return h

    def forward_noise(self, tilde_h):
        # z_pre will be used in the decoder cost
        z_pre = self.linear(tilde_h)
        z_pre_norm = self.bn_normalize(z_pre)
        # Add noise
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=z_pre_norm.size())
        noise = Variable(torch.FloatTensor(noise))
        # tilde_z will be used by decoder for reconstruction
        tilde_z = z_pre_norm + noise
        # store tilde_z in buffer
        self.buffer_tilde_z = tilde_z
        z = self.bn_gamma_beta(tilde_z)
        h = self.activation(z)
        return h


class StackedEncoders(torch.nn.Module):
    def __init__(self, d_in, d_encoders, activation_types,
                 train_batch_norms, noise_std):
        super(StackedEncoders, self).__init__()
        self.buffer_tilde_z_bottom = None
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        self.noise_level = noise_std
        n_encoders = len(d_encoders)
        for i in range(n_encoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_encoders[i - 1]
            d_output = d_encoders[i]
            activation = activation_types[i]
            train_batch_norm = train_batch_norms[i]
            encoder_ref = "encoder_" + str(i)
            encoder = Encoder(d_input, d_output, activation, train_batch_norm, noise_level=noise_std)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)

    def forward_clean(self, x):
        h = x
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=x.size())
        noise = Variable(torch.FloatTensor(noise))
        h = x + noise
        self.buffer_tilde_z_bottom = h.clone()
        # pass through encoders
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)
        return h

    def get_encoders_tilde_z(self, reverse=True):
        tilde_z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z.clone()
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers

    def get_encoders_z_pre(self, reverse=True):
        z_pre_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_z_pre.clone()
            z_pre_layers.append(z_pre)
        if reverse:
            z_pre_layers.reverse()
        return z_pre_layers

    def get_encoders_z(self, reverse=True):
        z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z = encoder.buffer_z.clone()
            z_layers.append(z)
        if reverse:
            z_layers.reverse()
        return z_layers