import numpy as np
from typing import *


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray = None
        self.creator: Function = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: List[Function] = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, gy):
        return gy * self.input.data * 2


def square(x: np.ndarray) -> Variable:
    return Square()(x)


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.input.data)


def exp(x: np.ndarray) -> Variable:
    return Exp()(x)


def numerical_diff(f: Function, x: Variable, eps=1e-4) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (eps * 2)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array([0.5, 2]))
y = square(exp(square(x)))
y.backward()
print(x.grad)
