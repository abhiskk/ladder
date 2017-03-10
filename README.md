# ladder

Implementation of [Ladder Network](https://arxiv.org/abs/1507.02672) in [PyTorch](http://pytorch.org/). 

### Requirements

- [PyTorch](http://pytorch.org/)

### Training ladder

1. Run ```python utils/mnist_data.py``` to create the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

2. Run the following command to train the *ladder* network:
  - ```python ladder/ladder.py --batch 100 --epochs 20 --noise_std 0.2 --data_dir data```

**Status**: The unsupervised loss starts at a high value because of which the network overfits the unsupervised loss and the supervised performance is bad.
