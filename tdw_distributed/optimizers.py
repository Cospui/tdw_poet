import numpy as np
import torch


class Optimizer(object):
    def __init__(self, theta):

        # print( "##### OPTIMIZER -> init @theta : ", theta )

        # self.dim is a list of zero. (float)
        self.dim = []

        for param_tensor in theta:
            layer_size = list(theta[param_tensor].size())
            # print( "##### OPTIMIZER -> init,  layer_size ", layer_size, " @ ", param_tensor )
            layer_zero = np.zeros(layer_size)
            self.dim.append(layer_zero)

        # self.dim = len(theta)
        # print( "##### OPTIMIZER -> init : ", self.dim )
        self.t = 0

    def update(self, theta, globalg, l2):
        self.t += 1
        step = self._compute_step(theta, globalg, l2)

        idx = 0
        ratio = []
        for param_tensor in theta:
            layer_norm = np.linalg.norm(theta[param_tensor].numpy())
            step_norm = np.linalg.norm(step[idx])
            ratio.append(step_norm / layer_norm)

            theta[param_tensor] = torch.from_numpy(theta[param_tensor].numpy() + step[idx]).double()
            idx += 1

        return ratio, theta

    def _compute_step(self, theta, globalg, l2):
        raise NotImplementedError


class SimpleSGD(Optimizer):
    def __init__(self, stepsize):
        self.stepsize = stepsize

    def compute(self, theta, globalg, l2):

        idx = 0
        for param_tensor in theta:
            layer_p = theta[param_tensor].numpy()
            layer_p = layer_p * l2
            globalg[idx] += layer_p
            idx += 1

        step = -self.stepsize * globalg

        idx = 0
        ratio = []
        for param_tensor in theta:
            layer_norm = np.linalg.norm(theta[param_tensor].numpy())
            step_norm = np.linalg.norm(step[idx])
            ratio.append(step_norm / layer_norm)

            theta[param_tensor] = torch.from_numpy(theta[param_tensor].numpy() + step[idx]).double()
            idx += 1

        return ratio, theta


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.9):
        Optimizer.__init__(self, theta)
        self.v = np.asarray(self.dim)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, theta, globalg, l2):
        idx = 0
        for param_tensor in theta:
            layer_p = theta[param_tensor].numpy()
            layer_p = layer_p * l2
            globalg[idx] += layer_p
            idx += 1

        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.init_stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.asarray(self.dim)
        self.v = np.asarray(self.dim)

    def reset(self):
        self.m = np.asarray(self.dim)
        self.v = np.asarray(self.dim)
        self.t = 0
        self.stepsize = self.init_stepsize

    def _compute_step(self, theta, globalg, l2):
        idx = 0
        for param_tensor in theta:
            layer_p = theta[param_tensor].numpy()
            layer_p = layer_p * l2
            globalg[idx] += layer_p
            idx += 1

        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * np.asarray(self.m)
        self.m += (1 - self.beta1) * np.asarray(globalg)
        self.v = self.beta2 * np.asarray(self.v) + (1 - self.beta2) * (globalg * globalg)
        # np.sqrt does not support ndarray
        for i in range(len(self.v)):
            self.v[i] = np.sqrt(self.v[i])
        step = -a * self.m / (self.v + self.epsilon)
        return step

    def propose(self, theta, globalg, l2):

        idx = 0
        for param_tensor in theta:
            layer_p = theta[param_tensor].numpy()
            layer_p = layer_p * l2
            globalg[idx] += layer_p
            idx += 1

        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        m = self.beta1 * self.m + (1 - self.beta1) * globalg
        v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)

        for i in range(len(self.v)):
            self.v[i] = np.sqrt(self.v[i])

        step = -a * m / (v + self.epsilon)
        idx = 0
        ratio = []
        for param_tensor in theta:
            layer_norm = np.linalg.norm(theta[param_tensor].numpy())
            step_norm = np.linalg.norm(step[idx])
            ratio.append(step_norm / layer_norm)

            theta[param_tensor] = torch.from_numpy(theta[param_tensor].numpy() + step[idx]).double()
            idx += 1

        return ratio, theta
