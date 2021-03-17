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


x = Variable(np.array([1, 2, 3]))
y = Square()(x)
print(y.data)
