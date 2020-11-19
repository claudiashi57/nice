import torch

class ChainEquationModel(object):
    def __init__(self, dim, scramble=False, hetero=False, setup='dag'):

        self.dim = dim // 2
        self.tdim = 1
        self.hetero = hetero
        self.setup = setup

        # init main relationships
        self.wxt = torch.randn(self.dim, self.dim) / dim
        self.wxy = torch.randn(self.dim, self.dim) / dim
        self.wyz = torch.randn(self.dim, self.dim) /dim

        if scramble:
            self.scramble, _ = torch.qr(torch.randn(dim+1, dim+1))
        else:
            self.scramble = torch.eye(dim+1)

    def __call__(self, n, env):

        x = torch.randn(n, self.dim) * env
        z = torch.randn(n, self.dim)
        "Generating the treatment t. P is the propensity score"
        p_val = x @ self.wxt + torch.randn(n, self.dim)
        p = torch.sigmoid(p_val.sum(1, keepdim=True))
        self.t = torch.bernoulli(p)
        """
        E[Y| X=x, T=t] = mu(x) + e*t + noise*env + tau(x)
        """
        e = self.dim * torch.ones_like(self.t)

        "heteroskedasticity among environments."
        if self.hetero:
            e_h = e + torch.randn(n, self.dim) * env
        else:
            e_h = e + torch.randn(n, self.dim)

        y_proto = x @ self.wxy + self.t * (e_h / self.dim) + torch.randn(n, self.dim)
        y_counter = x @ self.wxy + (torch.ones_like(self.t) - self.t) * (e_h / self.dim) + torch.randn(n, self.dim)

        if self.setup == 'dag':
            z = torch.rand(n, self.dim) * env

        elif self.setup == 'des':
            z = env *y_proto @ self.wyz + torch.randn(n, self.dim)

        elif self.setup == 'collider':
            z = env*y_proto @ self.wyz + self.t + torch.randn(n, self.dim)

        self.e_h = e_h
        self.e =e_h.mean()

        self.y = y_proto.sum(1, keepdim=True)
        self.y_cf = y_counter.sum(1, keepdim=True)
        self.x_input = torch.cat([x, z, torch.ones_like(self.t)], dim=1) @ self.scramble
        return self.x_input, self.t, p, self.y

    def sample_ite(self):
        return self.e_h

    def sample_att(self):
        treated = self.e_h[self.t==1]
        return torch.mean(treated)






