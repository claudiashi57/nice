#

import random

import icpy as icpy
import numpy as np
from sklearn.linear_model import LinearRegression
from torch.autograd import grad
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
from sem import *

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class IRM(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6
        x_val = environments[-1][0]
        t_val = environments[-1][1]
        y_val = environments[-1][3]
        self.x_all = torch.cat([x for (x, t, p, y) in environments])
        self.y_all = torch.cat([y for (x, t, p, y) in environments])
        self.t_all = torch.cat([t for (x, t, p, y) in environments])
        """
        validating on the third env
        """
        for reg in [1e-1]:
            "fixing the reg to be 1e-1 "
            self.train(environments[:-1], args, reg=reg)
            y_pred = self._phi(t_val, x_val) @ self.w
            err = (y_val - y_pred).pow(2).mean().item()
            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi0 = self.phi0.clone()
                best_phi1 = self.phi1.clone()

        self.phi0 = best_phi0
        self.phi1 = best_phi1

        self.y0 = self.x_all @ self.phi0 @ self.w
        self.y1 = self.x_all @ self.phi1 @ self.w

    def _phi(self, t, x):
        res = t * (x @ self.phi1) + (1 - t) * (x @ self.phi0)
        return res

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)
        self.phi0 = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.phi1 = torch.nn.Parameter(torch.eye(dim_x, dim_x))

        self.w = torch.ones(dim_x, 1)

        self.w.requires_grad = True
        loss = torch.nn.MSELoss()

        opt = torch.optim.Adam([self.phi1, self.phi0], lr=1e-3)

        for iteration in range(10000):
            penalty = 0
            e = 0
            for x_e, t_e, p_e, y_e in environments:

                error = loss(self._phi(t_e, x_e)@self.w, y_e)
                penalty += grad(error, self.w,
                                create_graph=True)[0].pow(2).mean()
                e += error
            opt.zero_grad()
            (reg * e + (1 - reg) * penalty).backward()
            opt.step()


    def ite(self):
        res = self.y1 - self.y0
        return res

    def accuracy(self):
        y_pred = self.y1 * self.t_all + self.y0 * (1 - self.t_all)
        y_true = self.y_all
        return (y_true - y_pred).pow(2).mean()

    def att(self):
        ites = self.ite().reshape(-1,1)
        return(ites[self.t_all==1].mean())



class LS2(object):
    def __init__(self, environments, args):
        x_all = torch.cat([x for (x, t, p, y) in environments]).numpy()
        y_all = torch.cat([y for (x, t, p, y) in environments]).numpy()
        t_all = torch.cat([t for (x, t, p, y) in environments]).numpy()
        x_all = x_all[:, :-1]
        t_all = t_all.squeeze()

        lg0 = LinearRegression().fit(x_all[t_all == 0], y_all[t_all == 0])
        lg1 = LinearRegression().fit(x_all[t_all == 1], y_all[t_all == 1])

        self.y_0 = lg0.predict(x_all)
        self.y_1 = lg1.predict(x_all)

        self.t_all = t_all.reshape(-1, 1)
        self.y_true = y_all

    def accuracy(self):
        return np.square(self.y_true - self.y_pred).mean()


    def ite(self):
        return self.y_1 - self.y_0

    def ate(self):
        return (self.ite()).mean()

    def att(self):
        ites = self.ite().reshape(-1,1)
        return(ites[self.t_all==1].mean())


class naive_estimater(object):
    def __init__(self, environments, args):
        self.y_all = torch.cat([y for (x, t, p, y) in environments]).numpy()
        self.t_all = torch.cat([t for (x, t, p, y) in environments]).numpy()

    def ate(self):
        y1 = (self.y_all * self.t_all).mean()
        y0 = (self.y_all * (1 - self.t_all)).mean()
        return y1 - y0


class ICP(object):
    def __init__(self, environments, args, alpha=0.1):
        x_all = torch.cat([x for (x, t, p, y) in environments]).numpy()
        y_all = torch.cat([y for (x, t, p, y) in environments]).numpy()
        t_all = torch.cat([t for (x, t, p, y) in environments]).numpy()
        x_all = x_all[:, :-1]
        t_all = t_all.squeeze()

        n = int(x_all.shape[0]/3)
        E = np.concatenate([np.ones((n)), np.ones((n)) * 2, np.ones((n)) * 3])

        s, q, p = icpy.invariant_causal_prediction(X=x_all, y=y_all.squeeze(), z=E, alpha=alpha)

        x_selected = x_all[:, s]
        # print("selected_percentage: {}".format(x_selected.shape[1]/x_all.shape[1]))

        if x_selected.shape[1] == 0:
            self.y_0 = np.ones(y_all.shape)
            self.y_1 = np.ones(y_all.shape)
        else:
            lg0 = LinearRegression().fit(x_selected[t_all == 0], y_all[t_all == 0])
            lg1 = LinearRegression().fit(x_selected[t_all == 1], y_all[t_all == 1])

            self.y_0 = lg0.predict(x_selected)
            self.y_1 = lg1.predict(x_selected)

        self.t_all = t_all.reshape(-1, 1)
        self.y_true = y_all

    def accuracy(self):
        return np.square(self.y_true - self.y_pred).mean()

    def ite(self):
        return self.y_1 - self.y_0

    def ate(self):
        return (self.ite()).mean()

    def att(self):
        ites = self.ite().reshape(-1, 1)
        return (ites[self.t_all == 1].mean())



