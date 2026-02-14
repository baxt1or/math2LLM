import math
import random
from sklearn.metrics import log_loss

random.seed(1244)

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1, 0, 0, 1]

w = [0.40682895453226675, 0.6857668949658486, -0.7095257369961039]
b = 0.33270629454892653

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dot_product(a, b):
    res = [sum([r * v for r, v in zip(row, b)]) for row in a]
    return res

def transpose(X):
    res = []
    n = len(X)
    m = len(X[0])
    for i in range(m):
        row = []
        for j in range(n):
            row.append(X[j][i])
        res.append(row)
    return res


lr = 0.01
grads = []

for i in range(100):

    pred = [sigmoid(v+b) for v in dot_product(xs, w)]
    errors = [p-y for p, y in zip(pred, ys)]

    loss = log_loss(ys, pred)

    dw = [v / len(xs) for v in dot_product(transpose(xs), errors)]
    db = sum(errors) / len(xs)

    grads.append(dw)

    w = [a-b for a, b in zip(w, [d * lr for d in dw])]
    b -= lr * db


final_loss = log_loss(ys, pred)
print(f"Final loss: {final_loss}")