from  __future__ import print_function

import sys
sys.path.append("/Users/abhishekkadian/Documents/Github/jaa-dl/assignment-1/")

import numpy as np

import torch
from torch.autograd import Variable
from torch.optim import SGD

import autoencoder as ae
import convnet


class SDA(torch.nn.Module):
    """Stacked Denoising Autoencoder

    reference: http://www.deeplearning.net/tutorial/SdA.html,
               http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
    """
    def __init__(self, d_input, d_hidden_autoencoders, d_out,
                 corruptions, batch_size, pre_lr=0.001, ft_lr=0.1):
        super(SDA, self).__init__()
        self.d_input = d_input
        self.d_hidden_autoencoders = list(d_hidden_autoencoders)
        self.d_out = d_out
        self.corruptions = corruptions
        self.batch_size = batch_size
        self.pre_lr = pre_lr
        self.ft_lr = ft_lr

        # Create one sequential module containing all autoencoders and logistic layer
        self.sequential = torch.nn.Sequential()

        # Create the Autoencoders
        self.autoencoders_ref = []
        for i, (d, c) in enumerate(zip(d_hidden_autoencoders, corruptions)):
            if i == 0:
                curr_input = d_input
            else:
                curr_input = d_hidden_autoencoders[i - 1]
            dna = ae.Autoencoder(curr_input, d, batch_size, corruption=c)
            self.autoencoders_ref.append("autoencoder_" + str(i))
            self.sequential.add_module(self.autoencoders_ref[-1], dna)

        # Create the Logistic Layer
        self.sequential.add_module("top_linear1", torch.nn.Linear(d_hidden_autoencoders[-1], d_out, bias=True))
        self.sequential.top_linear1 = torch.nn.Linear(d_hidden_autoencoders[-1], d_out, bias=True)
        self.sequential.top_linear1.weight.data = torch.zeros(self.sequential.top_linear1.weight.data.size())
        self.sequential.top_linear1.bias.data = torch.zeros(d_out)
        self.sequential.add_module("softmax", torch.nn.LogSoftmax())

    def pretrain(self, x, pt_epochs, verbose=True):
        n = x.data.size()[0]
        num_batches = n / self.batch_size
        t = x

        # Pre-train 1 autoencoder at a time
        for i, ae_re in enumerate(self.autoencoders_ref):
            # Get the current autoencoder
            ae = getattr(self.sequential, ae_re)

            # Getting encoded output from the previous autoencoder
            if i > 0:
                # Set the requires_grad to False so that backprop doesn't
                # travel all the way back to the previous autoencoder
                temp = Variable(torch.FloatTensor(n, ae.d_in), requires_grad=False)
                for k in range(num_batches):
                    start, end = k * self.batch_size, (k + 1) * self.batch_size
                    prev_ae = getattr(self.sequential, self.autoencoders_ref[i - 1])
                    temp.data[start:end] = prev_ae.encode(t[start:end], add_noise=False).data
                t = temp
            optimizer = SGD(ae.parameters(), lr=self.pre_lr)

            # Pre-training
            print("Pre-training Autoencoder:", i)
            for ep in range(pt_epochs):
                agg_cost = 0.
                for k in range(num_batches):
                    start, end = k * self.batch_size, (k + 1) * self.batch_size
                    bt = t[start:end]
                    optimizer.zero_grad()
                    z = ae.encode(bt, add_noise=True)
                    z = ae.decode(z)
                    loss = -torch.sum(bt * torch.log(z) + (1.0 - bt) * torch.log(1.0 - z), 1)
                    cost = torch.mean(loss)
                    cost.backward()
                    optimizer.step()
                    agg_cost += cost
                agg_cost /= num_batches
                if verbose:
                    print("Pre-training Autoencoder:", i, "Epoch:", ep, "Cost:", agg_cost.data[0])

    def forward(self, x):
        t = self.sequential.forward(x)
        return t

    def finetune(self, train_X, train_y, valid_X, valid_y,
                 valid_actual_size, ft_epochs, verbose=True):
        n = train_X.data.size()[0]
        num_batches = n / self.batch_size
        n_v = valid_X.data.size()[0]
        num_batches_v = n_v / self.batch_size
        optimizer = SGD(self.parameters(), lr=self.ft_lr)
        loss = torch.nn.NLLLoss()

        for ef in range(ft_epochs):
            agg_cost = 0
            for k in range(num_batches):
                start, end = k * self.batch_size, (k + 1) * self.batch_size
                bX = train_X[start:end]
                by = train_y[start:end]
                optimizer.zero_grad()
                p = self.forward(bX)
                cost = loss.forward(p, by)
                agg_cost += cost
                cost.backward()
                optimizer.step()
            agg_cost /= num_batches
            preds = np.zeros((n_v, self.d_out))

            # Calculate accuracy on Validation set
            for k in range(num_batches_v):
                start, end = k * self.batch_size, (k + 1) * self.batch_size
                bX = valid_X[start:end]
                p = self.forward(bX).data.numpy()
                preds[start:end] = p
            correct = 0
            for actual, prediction in zip(valid_y[:valid_actual_size], preds[:valid_actual_size]):
                ind = np.argmax(prediction)
                actual = actual.data.numpy()
                if ind == actual:
                    correct += 1

            if verbose:
                print("Fine-tuning Epoch:", ef, "Cost:", agg_cost.data[0],
                      "Validation Accuracy:", "{0:.4f}".format(correct / float(valid_actual_size)))


def main():
    # Load data
    trX, teX, trY, teY = convnet.load_mnist(onehot=False)
    trX = np.array([x.flatten() for x in trX])
    teX = np.array([x.flatten() for x in teX])
    trX = Variable(torch.from_numpy(trX).float())
    teX = Variable(torch.from_numpy(teX).float())
    trY = Variable(torch.from_numpy(trY).long())
    teY = Variable(torch.from_numpy(teY).long())

    batch_size = 64

    # Pad the validation set
    actual_size = teX.size()[0]
    padded_size = (actual_size / batch_size + 1) * batch_size
    teX_padded = Variable(torch.FloatTensor(padded_size, teX.size()[1]))
    teY_padded = Variable(torch.LongTensor(padded_size) * 0)
    teX_padded[:actual_size] = teX
    teY_padded[:actual_size] = teY

    sda = SDA(d_input=784,
              d_hidden_autoencoders=[1000, 1000, 1000],
              d_out=10,
              corruptions=[.1, .2, .3],
              batch_size=batch_size)

    sda.pretrain(trX, pt_epochs=15)

    sda.finetune(trX, trY, teX_padded, teY_padded,
                 valid_actual_size=actual_size, ft_epochs=36)


if __name__ == "__main__":
    main()