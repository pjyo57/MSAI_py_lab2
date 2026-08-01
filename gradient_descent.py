import numpy as np
def f(x):
    return (x-3) ** 2

def grad_f(x):
    return 2 * (x-3)

def gradient_descent_1d(start, lr, steps):

    x = start

    history = []

    for _ in range(steps):

        history.append(x)

        x = x - lr*grad_f(x)

    return x, np.array(history)


def f2(x,y):
    return x**2 + 5*y**2

def grad_f2(x,y):
    return 2*x, 10*y

def gradient_descent_2d(start, lr, steps):
    x, y = start
    path = [(x, y)]
    for _ in range(steps):
        dx, dy = grad_f2(x, y)
        x = x - lr * dx
        y = y - lr * dy
        path.append((x, y))
    return (x, y), np.array(path)


