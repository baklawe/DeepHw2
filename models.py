import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        lst = [self.in_features, *self.hidden_features]
        blocks = []
        # DONE: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        for idx in range(len(lst)-1):
            blocks.append(Linear(lst[idx], lst[idx+1], **kw))  # Moshe
            blocks.append(ReLU())
            if dropout is not 0:
                blocks.append(Dropout(p=dropout))
        blocks.append(Linear(lst[-1], self.num_classes, **kw))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        lst = [in_channels, *self.filters]
        layers = []
        # DONE: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        for idx in range(len(lst) - 1):
            layers.append(nn.Conv2d(lst[idx], lst[idx+1], kernel_size=(3, 3), padding=(1, 1)))
            layers.append(nn.ReLU())
            if (idx + 1) % self.pool_every is 0:
                layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2)))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        num_pool = int(len(self.filters) // self.pool_every)
        in_size = int(self.filters[-1] * (in_h // (2 ** num_pool)) * (in_w // (2 ** num_pool)))
        lst = [in_size, *self.hidden_dims]
        layers = []
        # DONE: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        for idx in range(len(lst)-1):
            layers.append(nn.Linear(lst[idx], lst[idx+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(lst[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # DONE: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        out = self.classifier(features.view([features.shape[0], -1]))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================

