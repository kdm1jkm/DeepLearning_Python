import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: np.ndarray = None


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
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


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.input.data)


def numerical_diff(f: Function, x: Variable, eps=1e-4) -> np.ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (eps * 2)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
