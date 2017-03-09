import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Decoder(torch.nn.Module):
    def __init__(self, d_in, d_out, use_cuda):
        super(Decoder, self).__init__()

        self.a1 = Parameter(0. * torch.ones(1, d_in))
        self.a2 = Parameter(1. * torch.ones(1, d_in))
        self.a3 = Parameter(0. * torch.ones(1, d_in))
        self.a4 = Parameter(0. * torch.ones(1, d_in))
        self.a5 = Parameter(0. * torch.ones(1, d_in))

        self.a6 = Parameter(0. * torch.ones(1, d_in))
        self.a7 = Parameter(1. * torch.ones(1, d_in))
        self.a8 = Parameter(0. * torch.ones(1, d_in))
        self.a9 = Parameter(0. * torch.ones(1, d_in))
        self.a10 = Parameter(0. * torch.ones(1, d_in))

        self.d_in = d_in
        self.d_out = d_out
        self.use_cuda = use_cuda

        if self.d_out is not None:
            self.V = torch.nn.Linear(d_in, d_out, bias=False)
            self.V.weight.data = torch.randn(self.V.weight.data.size()) / np.sqrt(d_in)
            # batch-normalization for u
            self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None

    def g(self, tilde_z_l, u_l):
        ones = Parameter(torch.ones(tilde_z_l.size()[0], 1))

        b_a1 = ones.mm(self.a1)
        b_a2 = ones.mm(self.a2)
        b_a3 = ones.mm(self.a3)
        b_a4 = ones.mm(self.a4)
        b_a5 = ones.mm(self.a5)

        b_a6 = ones.mm(self.a6)
        b_a7 = ones.mm(self.a7)
        b_a8 = ones.mm(self.a8)
        b_a9 = ones.mm(self.a9)
        b_a10 = ones.mm(self.a10)

        mu_l = torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u_l) + b_a3)) + \
               torch.mul(b_a4, u_l) + \
               b_a5

        v_l = torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u_l) + b_a8)) + \
              torch.mul(b_a9, u_l) + \
              b_a10

        hat_z_l = torch.mul(tilde_z_l - mu_l, v_l) + mu_l

        return hat_z_l

    def forward(self, tilde_z_l, u_l):
        # hat_z_l will be used for calculating decoder costs
        hat_z_l = self.g(tilde_z_l, u_l)
        # store hat_z_l in buffer for cost calculation
        self.buffer_hat_z_l = hat_z_l

        if self.d_out is not None:
            t = self.V.forward(hat_z_l)
            u_l_below = self.bn_normalize(t)
            return u_l_below
        else:
            return None


class StackedDecoders(torch.nn.Module):
    def __init__(self, d_in, d_decoders, image_size, use_cuda):
        super(StackedDecoders, self).__init__()
        self.bn_u_top = torch.nn.BatchNorm1d(d_in, affine=False)
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        self.use_cuda = use_cuda
        n_decoders = len(d_decoders)
        for i in range(n_decoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_decoders[i - 1]
            d_output = d_decoders[i]
            decoder_ref = "decoder_" + str(i)
            decoder = Decoder(d_input, d_output, use_cuda)
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)

        self.bottom_decoder = Decoder(image_size, None, use_cuda)

    def forward(self, tilde_z_layers, u_top, tilde_z_bottom):
        # Note that tilde_z_layers should be in reversed order of encoders
        hat_z = []
        u = self.bn_u_top(u_top)
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]
            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z_l)
        self.bottom_decoder.forward(tilde_z_bottom, u)
        hat_z_bottom = self.bottom_decoder.buffer_hat_z_l.clone()
        hat_z.append(hat_z_bottom)
        return hat_z

    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        # TODO: @Alex, @Joe review this
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            ones = Variable(torch.ones(z_pre.size()[0], 1))
            mean = torch.mean(z_pre, 0)
            var = np.var(z_pre.data.numpy(), axis=0).reshape(1, z_pre.size()[1])
            if self.use_cuda:
                var = Variable(torch.cuda.FloatTensor(var))
            else:
                var = Variable(torch.FloatTensor(var))
            hat_z_normalized = torch.div(hat_z - ones.mm(mean), ones.mm(torch.sqrt(var + 1e-5)))
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers_normalized