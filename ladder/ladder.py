from __future__ import print_function

import numpy as np
import argparse
import pickle
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from encoder import StackedEncoders
from decoder import StackedDecoders


class Ladder(torch.nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes, encoder_activations,
                 encoder_train_bn_scaling, noise_std, use_cuda):
        super(Ladder, self).__init__()
        self.use_cuda = use_cuda
        decoder_in = encoder_sizes[-1]
        encoder_in = decoder_sizes[-1]
        self.se = StackedEncoders(encoder_in, encoder_sizes, encoder_activations,
                                  encoder_train_bn_scaling, noise_std, use_cuda)
        self.de = StackedDecoders(decoder_in, decoder_sizes, encoder_in, use_cuda)
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)


def evaluate_performance(ladder, valid_loader, e, agg_cost_scaled, agg_supervised_cost_scaled,
                         agg_unsupervised_cost_scaled, args):
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        if args.cuda:
            output = output.cpu()
            target = target.cpu()
        output = output.data.numpy()
        preds = np.argmax(output, axis=1)
        target = target.data.numpy()
        correct += np.sum(target == preds)
        total += target.shape[0]

    print("Epoch:", e + 1, "\t",
          "Total Cost:", "{:.4f}".format(agg_cost_scaled), "\t",
          "Supervised Cost:", "{:.4f}".format(agg_supervised_cost_scaled), "\t",
          "Unsupervised Cost:", "{:.4f}".format(agg_unsupervised_cost_scaled), "\t",
          "Validation Accuracy:", correct / total)


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--u_costs", type=str, default="0.1, 0.1, 0.1, 0.1, 0.1, 10., 1000.")
    parser.add_argument("--cuda", type=bool, default=False)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    noise_std = args.noise_std
    data_dir = args.data_dir
    seed = args.seed
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.\n")
        args.cuda = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
    print("NOISE STD:", noise_std)
    print("CUDA:", args.cuda)
    print("=====================\n")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_labelled_images_filename = os.path.join(args.data_dir, "train_labelled_images.p")
    train_labelled_labels_filename = os.path.join(args.data_dir, "train_labelled_labels.p")
    train_unlabelled_images_filename = os.path.join(args.data_dir, "train_unlabelled_images.p")
    train_unlabelled_labels_filename = os.path.join(args.data_dir, "train_unlabelled_labels.p")
    validation_images_filename = os.path.join(args.data_dir, "validation_images.p")
    validation_labels_filename = os.path.join(args.data_dir, "validation_labels.p")

    print("Loading Data")
    with open(train_labelled_images_filename) as f:
        train_labelled_images = pickle.load(f)
    train_labelled_images = train_labelled_images.reshape(train_labelled_images.shape[0], 784)
    with open(train_labelled_labels_filename) as f:
        train_labelled_labels = pickle.load(f).astype(int)
    with open(train_unlabelled_images_filename) as f:
        train_unlabelled_images = pickle.load(f)
    train_unlabelled_images = train_unlabelled_images.reshape(train_unlabelled_images.shape[0], 784)
    with open(train_unlabelled_labels_filename) as f:
        train_unlabelled_labels = pickle.load(f).astype(int)
    with open(validation_images_filename) as f:
        validation_images = pickle.load(f)
    validation_images = validation_images.reshape(validation_images.shape[0], 784)
    with open(validation_labels_filename) as f:
        validation_labels = pickle.load(f).astype(int)

    # Create DataLoaders
    unlabelled_dataset = TensorDataset(torch.FloatTensor(train_unlabelled_images), torch.LongTensor(train_unlabelled_labels))
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    validation_dataset = TensorDataset(torch.FloatTensor(validation_images), torch.LongTensor(validation_labels))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Configure the Ladder
    encoder_sizes = [1000, 500, 250, 250, 250, 10]
    decoder_sizes = [250, 250, 250, 500, 1000, 784]
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]
    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "softmax"]
    encoder_train_bn_scaling = [False, False, False, False, False, True]
    ladder = Ladder(encoder_sizes, decoder_sizes, encoder_activations,
                    encoder_train_bn_scaling, noise_std, args.cuda)
    optimizer = Adam(ladder.parameters(), lr=0.002)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    if args.cuda:
        ladder.cuda()

    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)

    print("")
    print("========NETWORK=======")
    print(ladder)
    print("======================")

    print("")
    print("==UNSUPERVISED-COSTS==")
    print(unsupervised_costs_lambda)

    print("")
    print("=====================")
    print("TRAINING\n")

    # TODO: Add annealing of learning rate after 100 epochs

    for e in range(epochs):
        agg_cost = 0.
        agg_supervised_cost = 0.
        agg_unsupervised_cost = 0.
        num_batches = 0
        ladder.train()
        # TODO: Add volatile for the input parameters in training and validation
        ind_labelled = 0
        ind_limit = np.ceil(float(train_labelled_images.shape[0]) / batch_size)

        for batch_idx, (unlabelled_images, unlabelled_labels) in enumerate(unlabelled_loader):
            if ind_labelled == ind_limit:
                randomize = np.arange(train_labelled_images.shape[0])
                np.random.shuffle(randomize)
                train_labelled_images = train_labelled_images[randomize]
                train_labelled_labels = train_labelled_labels[randomize]
                ind_labelled = 0

            labelled_start = batch_size * ind_labelled
            labelled_end = batch_size * (ind_labelled + 1)
            ind_labelled += 1
            batch_train_labelled_images = torch.FloatTensor(train_labelled_images[labelled_start:labelled_end])
            batch_train_labelled_labels = torch.LongTensor(train_labelled_labels[labelled_start:labelled_end])

            if args.cuda:
                batch_train_labelled_images = batch_train_labelled_images.cuda()
                batch_train_labelled_labels = batch_train_labelled_labels.cuda()
                unlabelled_images = unlabelled_images.cuda()

            labelled_data = Variable(batch_train_labelled_images, requires_grad=False)
            labelled_target = Variable(batch_train_labelled_labels, requires_grad=False)
            unlabelled_data = Variable(unlabelled_images)

            optimizer.zero_grad()

            # do a noisy pass for labelled data
            output_noise_labelled = ladder.forward_encoders_noise(labelled_data)

            # do a noisy pass for unlabelled_data
            output_noise_unlabelled = ladder.forward_encoders_noise(unlabelled_data)
            tilde_z_layers_unlabelled = ladder.get_encoders_tilde_z(reverse=True)

            # do a clean pass for unlabelled data
            output_clean_unlabelled = ladder.forward_encoders_clean(unlabelled_data)
            z_pre_layers_unlabelled = ladder.get_encoders_z_pre(reverse=True)
            z_layers_unlabelled = ladder.get_encoders_z(reverse=True)

            tilde_z_bottom_unlabelled = ladder.get_encoder_tilde_z_bottom()

            # pass through decoders
            hat_z_layers_unlabelled = ladder.forward_decoders(tilde_z_layers_unlabelled,
                                                              output_noise_unlabelled,
                                                              tilde_z_bottom_unlabelled)

            z_pre_layers_unlabelled.append(unlabelled_data)
            z_layers_unlabelled.append(unlabelled_data)

            # TODO: Verify if you have to batch-normalize the bottom-most layer also
            # batch normalize using mean, var of z_pre
            bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(hat_z_layers_unlabelled, z_pre_layers_unlabelled)

            # calculate costs
            cost_supervised = loss_supervised.forward(output_noise_labelled, labelled_target)
            cost_unsupervised = 0.
            assert len(z_layers_unlabelled) == len(bn_hat_z_layers_unlabelled)
            for cost_lambda, z, bn_hat_z in zip(unsupervised_costs_lambda, z_layers_unlabelled, bn_hat_z_layers_unlabelled):
                c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
                cost_unsupervised += c

            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()
            optimizer.step()

            agg_cost += cost.data[0]
            agg_supervised_cost += cost_supervised.data[0]
            agg_unsupervised_cost += cost_unsupervised.data[0]
            num_batches += 1

            if ind_labelled == ind_limit:
                # Evaluation
                ladder.eval()
                evaluate_performance(ladder, validation_loader, e,
                                     agg_cost / num_batches,
                                     agg_supervised_cost / num_batches,
                                     agg_unsupervised_cost / num_batches,
                                     args)
                ladder.train()
    print("=====================\n")
    print("Done :)")


if __name__ == "__main__":
    main()
