import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
import numpy as np
from chainer import cuda
xp = cuda.cupy
class MnistMLP(chainer.Chain):
    def __init__(self, layer_sizes, train = True) :
        super(MnistMLP, self).__init__()
        self.size = len(layer_sizes) - 1
        for idx in range(self.size) :
            self.add_link("l{0}".format(idx + 1),
                L.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        for idx in range(self.size - 1):
            self.add_link("b{0}".format(idx + 1),
                L.Linear(layer_sizes[idx+1], layer_sizes[idx]))
        self.lambdas = None
        self.y = None
        self.loss = None
        self.accuracy = None
        self.pretrain = True
        self.train=train
 
    def __call__(self, x, t, j) :
        self.clear()
        return self.autoencoder(x,t,j)

    def autoencoder(self, x, t, j) :
        h = x
        hb = None
        idx = 0
        for idx in range(j) :
            if idx == j-1 and self.pretrain:
                h = Variable(xp.asarray(h.data, dtype=xp.float32))
            hb = h
            if idx == self.size-1: break
            h = F.sigmoid(self.__getitem__("l{0}".format(idx + 1))(h))
        if self.pretrain:
            d = F.sigmoid(self.__getitem__("b{0}".format(idx + 1))(h))
            self.loss = F.mean_squared_error(hb, d)
        else:
            self.y = self.__getitem__("l{0}".format(idx + 1))(h)
            self.accuracy = accuracy.accuracy(self.y, t)
            self.loss = F.softmax_cross_entropy(self.y,t)
        return self.loss

    def setfinetuning(self):
        self.pretrain = False

    def clear(self) :
        self.y = None
        self.accuracy = None
        self.loss = None


class MnistMLPParallel(chainer.Chain):

    """An example of model-parallel MLP.

    This chain combines four small MLPs on two different devices.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLPParallel, self).__init__(
            first0=MnistMLP(n_in, n_units // 2, n_units).to_gpu(0),
            first1=MnistMLP(n_in, n_units // 2, n_units).to_gpu(1),
            second0=MnistMLP(n_units, n_units // 2, n_out).to_gpu(0),
            second1=MnistMLP(n_units, n_units // 2, n_out).to_gpu(1),
        )

    def __call__(self, x):
        # assume x is on GPU 0
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1(x1)

        # sync
        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(h1))

        # sync
        y = y0 + F.copy(y1, 0)
        return y
