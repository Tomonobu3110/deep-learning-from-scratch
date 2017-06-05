import numpy as np

X = np.array([0.6, 0.9])      # input
T = np.array([1.0, 0.0, 1.0]) # teacher
W = np.random.randn(2, 3)     # weight

def predict(x, w):
    return np.dot(x, w)

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def process_all(x, w, t):
    Y    = predict(x, w) # output
    NY   = softmax(Y)    # normalized output
    return cross_entropy_error(NY, t) # loss

def numerical_gradient(x, w, t):
    h = 1e-4
    grad = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        WdM = np.array(w) # Weight (delta Minus)
        WdP = np.array(w) # Wegith (delta Plus)
        WdM[idx] -= h
        WdP[idx] += h

        lossM = process_all(x, WdM, t)
        lossP = process_all(x, WdP, t)

        grad[idx] = (lossP - lossM) / (2 * h)
        it.iternext()

    return grad


