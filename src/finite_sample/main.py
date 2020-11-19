import argparse

import torch
import torch.nn.functional as F
from gen_dat import *
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='scm')
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--net', type=str, default='dragon')
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--d', type=int, default=30)
parser.add_argument('--setup', type=str, default='x_all')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alltrain', type=int, default=0)
parser.add_argument('--shuffle', type=int, default=0)
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

error = []
adjust_none = []


def make_env(flags, ep, i):
    print(i)
    setup = flags.setup
    df = np.load('../../dat/exp3/replication_{}_ep_{}.npz'.format(i, ep))

    X_all = np.concatenate([df['A'], df['X'], df['Z']], axis=1)

    pre_x = df['A']
    Y = df["Y"]
    T = df["T"]
    ITE = df["ITE"]
    print("the naive estimate should be {}".format(Y[T == 1].mean() - Y[T == 0].mean()))

    if setup == 'x_and_a':
        print('use all variables that are safe to adjust for ')
        X = np.concatenate([df['X'], df['A']], axis=1)

    elif setup == 'x_all':
        X = X_all
        print('use all the data: {}'.format(setup))
        print(X.shape)

    elif setup == 'xt':
        X = df['pre_treatment']

    elif setup == 'xtat':
        X = np.concatenate([df['pre_treatment'], pre_x[:, 10:22]], axis=1)
    else:
        import ipdb;
        ipdb.set_trace()

    return {
        'covariates': torch.from_numpy(X),
        'outcome': torch.from_numpy(Y),
        'treatment': torch.from_numpy(T),
        'ITE_oracle': torch.from_numpy(ITE)
    }


def shuffle_env(env, envs, index):
    new_env = env.copy()
    for k in env.keys():
        new_env[k] = torch.cat([envs[0][k][index], envs[1][k][index], envs[2][k][index]])
    return new_env


for i in range(5):
    torch.manual_seed(i)
    envs = [make_env(flags, 0.2, i),
            make_env(flags, 1, i),
            make_env(flags, 5, i)]
    if flags.shuffle == 1:
        n = int(envs[0]['outcome'].shape[0])

        index = np.random.choice(n, n, replace=False)
        i0 = index[:int(n / 3)]
        i1 = index[int(n / 3): int(2 * n / 3)]
        i2 = index[int(2 * n / 3):]
        new_envs = [
            shuffle_env(envs[0], envs, i0),
            shuffle_env(envs[1], envs, i1),
            shuffle_env(envs[2], envs, i2)
        ]
        envs = new_envs
    else:
        envs = envs

    # # Define and instantiate the model
    class MLP(nn.Module):
        def __init__(self, dim):
            super(MLP, self).__init__()
            hidden_dim = flags.hidden_dim
            hypo_dim = 100
            self.lin1 = nn.Linear(dim, hidden_dim)
            self.lin1_1 = nn.Linear(hidden_dim, hidden_dim)
            self.lin1_2 = nn.Linear(hidden_dim, hypo_dim)
            # tarnet
            if flags.net == 'tarnet':
                self.lin1_3 = nn.Linear(dim, 1)
            else:
                self.lin1_3 = nn.Linear(hypo_dim, 1)

            self.lin2_0 = nn.Linear(hypo_dim, hypo_dim)
            self.lin2_1 = nn.Linear(hypo_dim, hypo_dim)

            self.lin3_0 = nn.Linear(hypo_dim, hypo_dim)
            self.lin3_1 = nn.Linear(hypo_dim, hypo_dim)

            self.lin4_0 = nn.Linear(hypo_dim, 1)
            self.lin4_1 = nn.Linear(hypo_dim, 1)

            for lin in [self.lin1, self.lin1_1, self.lin1_2, self.lin2_0, self.lin2_1, self.lin1_3, self.lin3_0,
                        self.lin3_1, self.lin4_0, self.lin4_1]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)

        def forward(self, input):
            initial = input.view(input.shape)

            x = F.relu(self.lin1(initial))
            x = F.relu(self.lin1_1(x))
            x = F.relu(self.lin1_2(x))
            x = F.relu(x)
            if flags.net == 'tarnet':
                t = self.lin1_3(initial)
            else:
                t = self.lin1_3(x)

            h1 = F.relu(self.lin2_1(x))
            h0 = F.relu(self.lin2_0(x))

            h1 = F.relu(h1)
            h0 = F.relu(h0)

            h0 = F.relu(self.lin3_0(h0))
            h1 = F.relu(self.lin3_1(h1))

            h0 = self.lin4_0(h0)
            h1 = self.lin4_1(h1)
            return torch.cat((h0, h1, t), 1)


    mlp = MLP(envs[0]['covariates'].shape[1])
    if flags.gpu == 1:
        mlp.cuda()
    else:
        mlp

    # Define loss function helpers

    def mean_nll(y_logit, y):
        return nn.functional.binary_cross_entropy_with_logits(y_logit, y.float())


    def mean_accuracy(y_logit, y):
        preds = (y_logit > 0.).double()
        return ((preds - y).abs() < 1e-2).float().mean()


    def penalty(y_logit, y):
        if flags.gpu == 1:
            scale = torch.tensor(1.).cuda().requires_grad_()
        else:
            scale = torch.tensor(1.).requires_grad_()
        loss = mean_nll(y_logit * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        res = torch.sum(grad ** 2)
        return res


    def ite(y0_logit, y1_logit):
        y0_pred = torch.sigmoid(y0_logit).float()
        y1_pred = torch.sigmoid(y1_logit).float()
        return y1_pred - y0_pred


    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))


    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'pred_att', 'true_att', 'train_t_acc')


    def att(ite, t):
        ite = np.concatenate(ite)
        t = np.concatenate(t)
        return np.mean(ite[t == 1])


    def train_stack(val, env, all_train=True):
        return torch.stack([env[0][val], env[1][val], env[2][val]]).mean()


    # training loop
    optimizer_adam = optim.Adam(mlp.parameters(), lr=flags.lr)
    optimizer_sgd = optim.SGD(mlp.parameters(), lr=1e-7, momentum=0.9)

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['covariates'].float())
            y0_logit = logits[:, 0].unsqueeze(1)
            y1_logit = logits[:, 1].unsqueeze(1)
            t_logit = logits[:, 2].unsqueeze(1)
            t = env['treatment'].float()
            y_logit = t * y1_logit + (1 - t) * y0_logit

            env['ite'] = ite(y0_logit, y1_logit)

            env['nll'] = mean_nll(y_logit, env['outcome'])
            env['t_nll'] = mean_nll(t_logit, t)

            env['acc'] = mean_accuracy(y_logit, env['outcome'])
            env['t_acc'] = mean_accuracy(t_logit, env['treatment'])

            env['penalty'] = penalty(y_logit, env['outcome'])

        all_train = True
        train_nll = train_stack('nll', envs, all_train=all_train)
        train_t_nll = train_stack('t_nll', envs, all_train=all_train)
        train_acc = train_stack('acc', envs, all_train=all_train)
        train_t_acc = train_stack('t_acc', envs, all_train=all_train)

        train_ate = train_stack('ite', envs, all_train=all_train)
        train_penalty = train_stack('penalty', envs, all_train=all_train)
        if flags.gpu == 1:
            weight_norm = torch.tensor(0.).cuda()
        else:
            weight_norm = torch.tensor(0.)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()

        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
                          if step >= flags.penalty_anneal_iters else 1)
        loss += penalty_weight * train_penalty
        loss += train_t_nll
        # loss = train_t_nll.clone()

        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        all_t = torch.stack([envs[0]['treatment'], envs[1]['treatment'], envs[2]['treatment']]).detach().cpu().numpy()
        all_y = torch.stack([envs[0]['outcome'], envs[1]['outcome'], envs[2]['outcome']]).detach().cpu().numpy()

        pred_ite = torch.stack([envs[0]['ite'], envs[1]['ite'], envs[2]['ite']]).detach().cpu().numpy()
        true_ite = torch.stack(
            [envs[0]['ITE_oracle'], envs[1]['ITE_oracle'], envs[2]['ITE_oracle']]).detach().cpu().numpy()

        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy() + train_t_nll.detach().cpu().numpy(),

                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                att(pred_ite, all_t),
                att(true_ite, all_t),
                train_t_acc.detach().cpu().numpy(),
            )
    # import ipdb; ipdb.set_trace()
    err = abs(att(pred_ite, all_t) - att(true_ite, all_t))
    no_adjust = all_y[all_t == 1].mean() - all_y[all_t == 0].mean()

    unadjusted_err = abs(no_adjust - att(true_ite, all_t))

    error.append(err)
    adjust_none.append(unadjusted_err)

if flags.shuffle == 1:
    s = 'shuffle'
else:
    s = 'no_shuffle'

if flags.penalty_weight > 0:
    mod = 'irm'
if flags.penalty_weight == 0:
    mod = 'erm'

saver = pd.DataFrame(error)
no_adjust = pd.DataFrame(adjust_none)
print('noadjust is: {}'.format(adjust_none))

saver.to_csv("../../res/exp3/{}_{}_{}.csv".format(s, mod, flags.setup), index=False)
no_adjust.to_csv("../../res/exp3/{}_no_adjustment_{}_{}.csv".format(s,mod, flags.setup), index=False)
