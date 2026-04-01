import random
from micrograd.engine import Value
from typing import List

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, inp_dim:int, nonlin=True):
        self.inp_dim = inp_dim
        self.non_lin = nonlin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(self.inp_dim)]
        self.b = Value(0)

    def __call__(self, x:List): # x should be a list with length = inp_dim ie input for each weight
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.non_lin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, din, dout, **kwargs):
        self.neurons = [Neuron(din, **kwargs) for _ in range(dout)]

    def __call__(self, x:List): # should be list with length = inp_dim
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, din, douts):
        self.sz = [din] + douts
        self.layers = [
            Layer(self.sz[i], self.sz[i + 1], nonlin=i != len(douts) - 1)
            for i in range(len(douts))
        ]

    def __call__(self, x:List): # x should be list of len inp_dim = one value for each of the neuron for first layer
        # so for each data point we must have a 2d list, ie a list of list of x-values
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
