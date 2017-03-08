from __future__ import print_function
import os
import urllib
import random
import numpy as np
import gzip
from collections import defaultdict
import pickle


def get_data(filename, directory,
             data_url="http://yann.lecun.com/exdb/mnist/",
             verbose=True):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(data_url + filename, filepath)
        statinfo = os.stat(filepath)
        if verbose:
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename, verbose=True):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    if verbose:
        print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
              'Invalid magic number %d in MNIST image file: %s' %
              (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename, verbose=True):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    if verbose:
        print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
              'Invalid magic number %d in MNIST label file: %s' %
              (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def shuffle_images_labels(images, labels):
    assert images.shape[0] == labels.shape[0]
    randomize = np.arange(images.shape[0])
    np.random.shuffle(randomize)
    return images[randomize], labels[randomize]


def dump_pickle(filepath, d):
    with open(filepath, "wb") as f:
        pickle.dump(d, f)


def main():
    n_labelled = 3000
    random.seed(42)
    np.random.seed(42)
    data_dir = "data/"
    mnist_train_images_gz = 'train-images-idx3-ubyte.gz'
    mnist_train_labels_gz = 'train-labels-idx1-ubyte.gz'
    mnist_test_images_gz = 't10k-images-idx3-ubyte.gz'
    mnist_test_labels_gz = 't10k-labels-idx1-ubyte.gz'

    mnist_train_images = get_data(mnist_train_images_gz, data_dir)
    mnist_train_images = extract_images(mnist_train_images)
    mnist_train_labels = get_data(mnist_train_labels_gz, data_dir)
    mnist_train_labels = extract_labels(mnist_train_labels)
    mnist_test_images = get_data(mnist_test_images_gz, data_dir)
    mnist_test_images = extract_images(mnist_test_images)
    mnist_test_labels = get_data(mnist_test_labels_gz, data_dir)
    mnist_test_labels = extract_labels(mnist_test_labels)

    train_data_shuffle = [(x, y) for x, y in zip(mnist_train_images, mnist_train_labels)]
    random.shuffle(train_data_shuffle)
    mnist_shuffled_train_images = np.array([x[0] for x in train_data_shuffle])
    mnist_shuffled_train_labels = np.array([x[1] for x in train_data_shuffle])

    validation_size = 10000
    train_size = mnist_train_images.shape[0] - validation_size

    train_images = mnist_shuffled_train_images[:train_size].copy()
    train_labels = mnist_shuffled_train_labels[:train_size].copy()

    validation_images = mnist_shuffled_train_images[train_size:].copy()
    validation_labels = mnist_shuffled_train_labels[train_size:].copy()

    test_images = mnist_test_images
    test_labels = mnist_test_labels

    train_data_label_buckets = defaultdict(list)

    for image, label in zip(train_images, train_labels):
        train_data_label_buckets[label].append((image, label))

    num_labels = len(train_data_label_buckets)

    train_labelled_data_images = []
    train_labelled_data_labels = []
    train_unlabelled_data_images = []
    train_unlabelled_data_labels = []

    for label, label_data in train_data_label_buckets.items():
        count = n_labelled / num_labels
        for v in label_data[:count]:
            train_labelled_data_images.append(v[0])
            train_labelled_data_labels.append(v[1])
        for v in label_data[count:]:
            train_unlabelled_data_images.append(v[0])
            # dummy label
            train_unlabelled_data_labels.append(-1)

    train_labelled_images = np.array(train_labelled_data_images)
    train_labelled_labels = np.array(train_labelled_data_labels)

    train_unlabelled_images = np.array(train_unlabelled_data_images)
    train_unlabelled_labels = np.array(train_unlabelled_data_labels)

    train_labelled_images = train_labelled_images[:, :, :, 0]
    train_unlabelled_images = train_unlabelled_images[:, :, :, 0]
    validation_images = validation_images[:, :, :, 0]
    test_images = test_images[:, :, :, 0]

    train_labelled_images, train_labelled_labels = shuffle_images_labels(train_labelled_images, train_labelled_labels)

    # normalizing
    train_labelled_images = np.multiply(train_labelled_images, 1./255.)
    train_unlabelled_images = np.multiply(train_unlabelled_images, 1./255.)
    validation_images = np.multiply(validation_images, 1./255.)
    test_images = np.multiply(test_images, 1./255,)

    print("=" * 50)
    print("train_labelled_images shape:", train_labelled_images.shape)
    print("train_labelled_labels shape:", train_labelled_labels.shape)
    print()
    print("train_unlabelled_images shape:", train_unlabelled_images.shape)
    print("train_unlabelled_labels shape:", train_unlabelled_labels.shape)
    print()
    print("validation_images shape:", validation_images.shape)
    print("validation_labels shape:", validation_labels.shape)
    print()
    print("test_images shape:", test_images.shape)
    print("test_labels shape:", test_labels.shape)
    print("=" * 50)

    print("Dumping pickles")

    dump_pickle(data_dir + "train_labelled_images.p", train_labelled_images)
    dump_pickle(data_dir + "train_labelled_labels.p", train_labelled_labels)
    dump_pickle(data_dir + "train_unlabelled_images.p", train_unlabelled_images)
    dump_pickle(data_dir + "train_unlabelled_labels.p", train_unlabelled_labels)
    dump_pickle(data_dir + "validation_images.p", validation_images)
    dump_pickle(data_dir + "validation_labels.p", validation_labels)
    dump_pickle(data_dir + "test_images.p", test_images)
    dump_pickle(data_dir + "test_labels.p", test_labels)

    print("MNIST dataset successfully created")

if __name__ == "__main__":
    main()