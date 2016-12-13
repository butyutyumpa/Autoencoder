import chainer.functions as F
import numpy as np
import chainer
from chainer import serializers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import net
import pylab
import math
#model = cPickle.load(open("model.pkl", "rb"))

def draw_digit_w1(data, n, i, length):
    size = 28
    pylab.subplot(28, 28, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    pylab.xlim(0,size)
    pylab.ylim(0,size)
    pylab.pcolor(Z)
    pylab.title("%d"%i, size=9)
    pylab.gray()
    pylab.tick_params(labelbottom="off")
    pylab.tick_params(labelleft="off")


lsizes = [784,50,50,10]
model = net.MnistMLP(layer_sizes = lsizes)
serializers.load_npz('mlp.model', model)
layer = model.__getitem__("l1")
pylab.style.use('fivethirtyeight')
pylab.figure(figsize=(28,28))
cnt = 1
for i in range(len(layer.W.data)):
    draw_digit_w1(layer.W.data[i], cnt, i, layer.W.data[9].size)
    cnt += 1

#pylab.imshow(layer.W.data)
#pylab.gray()
pylab.savefig('layer1.png')

