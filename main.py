import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    def forward(self, x: np.ndarray):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)


def numerical_diff(f: Function, x: Variable, eps=1e-4) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (eps * 2)


x = Variable(np.array(2.0))
dy = numerical_diff(Exp(), x)
print(dy)
print(Exp()(x).data)
