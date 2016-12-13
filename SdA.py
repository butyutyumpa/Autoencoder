#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer import functions as F
import data
import net
import csv
import sys

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--pepoch', '-p', default=15, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=50, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
p_epoch = args.pepoch
n_units = args.unit


print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('')

def savecsv(lst, path):
    dataWriter = csv.writer(open(path, 'w'))
    dataWriter.writerow(lst)

# Prepare dataset
def train(trainL, trainA, testL, testA):
    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []
    l1w=[]
    l2w=[]
    l3w=[]
    print('load MNIST dataset')
    mnist = data.load_mnist_data()
    mnist['data'] = mnist['data'].astype(np.float32)
    mnist['data'] /= 255
    mnist['target'] = mnist['target'].astype(np.int32)

    N = 1000
    lsizes=[784,50,50,10]
    x_train, x_test = np.split(mnist['data'],   [N])
    y_train, y_test = np.split(mnist['target'], [N])
    N_test = y_test.size

    # Prepare multi-layer perceptron model, defined in net.py
    if args.net == 'simple':
        #model = net.MnistMLP(lsizes)
        model = net.MnistMLP(layer_sizes = lsizes)
        if args.gpu >= 0:
            cuda.get_device(args.gpu).use()
            model.to_gpu()
        xp = np if args.gpu < 0 else cuda.cupy
    elif args.net == 'parallel':
        cuda.check_cuda_available()
        model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
        xp = cuda.cupy
    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)
    # Pretrain loop
    print("start pretrain")
    epo = p_epoch
    for j in six.moves.range(1, len(lsizes)):
        if j == len(lsizes)-1:
            model.setfinetuning()
            print("start finetuning")
            epo = n_epoch
        for epoch in six.moves.range(1, epo + 1):
            print('layer ', j ,'p_epoch ', epoch)
            perm = np.random.permutation(N)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, N, batchsize):
                x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))
                optimizer.update(model, x, t, j)
                sum_loss += float(model.loss.data) * len(t.data)
                if not model.pretrain:
                    sum_accuracy += float(model.accuracy.data) * len(t.data)
            if model.pretrain:
                print('Pretrain: train mean loss={}'.format(
                         sum_loss / N))
            else:
                print('Finetune: train mean loss={}, accuracy={}'.format(
                         sum_loss / N, sum_accuracy / N))
            trainLoss.append(sum_loss/N)
            trainAccuracy.append(sum_accuracy/N)
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            model.train=False
            for i in six.moves.range(0, N_test, batchsize):
                x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                     volatile='on')
                t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                                     volatile='on')
                loss = model(x, t, j)
                sum_loss += float(loss.data) * len(t.data)
                if not model.pretrain:
                    sum_accuracy += float(model.accuracy.data) * len(t.data)
            if model.pretrain:
                print('Pretrain: test  mean loss={}'.format(
                    sum_loss / N_test))
            else:
                print('Finetune: test  mean loss={}, accuracy={}'.format(
                    sum_loss / N_test, sum_accuracy / N_test))
            testLoss.append(sum_loss/N_test)
            testAccuracy.append(sum_accuracy/N_test)
            model.train=True

    # Save the model and the optimizer
    savecsv(trainLoss, trainL)
    savecsv(trainAccuracy, trainA)
    savecsv(testLoss, testL)
    savecsv(testAccuracy, testA)

    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)

if __name__ == "__main__":
    for i in xrange(1):
        train("result/trainLoss{}.csv".format(i),
                    "result/trainAccuracy{}.csv".format(i),
                    "result/testLoss{}.csv".format(i),
                    "result/testAccuracy{}.csv".format(i))

