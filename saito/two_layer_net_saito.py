import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# maybe x is batch
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

# maybe y and t are batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

# f is function, param is Wx or bx
def numerical_gradient(f, param):
    h = 1e-4
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index # (0,0) -> (0,1) -> ... (1,0) ... (n, m)
        tmp_val = param[idx]

        param[idx] = float(tmp_val) + h
        fxh1 = f(param) # f(x+h) 

        param[idx] = float(tmp_val) - h
        fxh2 = f(param) # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        param[idx] = tmp_val # restoer value
        it.iternext()

    return grad

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # init params
        self.params = {}
        # input to hidden
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # hidden to outpu
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        # params
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        # predict from input to hidden
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # predict from hidden to output
        a2 = np.dot(z1, W2) + b2
        y  = softmax(a2)

        return y

    # x: input data / t: teacher data (maybe batch)
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return arruracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


