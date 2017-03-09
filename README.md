# ladder

Implementation of [Ladder Network](https://arxiv.org/abs/1507.02672) in [PyTorch](http://pytorch.org/). 

### Requirements

- [PyTorch](http://pytorch.org/)

### Training ladder

1. Run ```python utils/mnist_data.py``` to create the dataset which will create the ```data``` folder

2. Run ```python ladder/ladder.py --batch 100 --epochs 150 --noise_std 0.2``` to train the *ladder* network.

**Status**: The unsupervised loss starts at a high value because of which the network overfits the unsupervised loss and the supervised performance is bad.
