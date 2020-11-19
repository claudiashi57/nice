import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='SpeedDATING')
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--env', type=int, default=22)
parser.add_argument('--collider', type=int, default=1)
parser.add_argument('--num_col', type=int, default=0)
parser.add_argument('--mod', type=str, default='mod4')
parser.add_argument('--dimension', type=str, default='high')
parser.add_argument('--dat', type=int, default=1)
parser.add_argument('--net', type=str, default='tarnet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--icp', type=int, default=1)
parser.add_argument('--all_train', type=int, default=0)
parser.add_argument('--data_base_dir', type=str, default='')
parser.add_argument('--output_base_dir', type=str, default='')

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []

final_train_ate = []
final_test_ate = []
final_train_treatment_acc=[]
final_test_treatment_acc=[]

torch.manual_seed(1)

# Load data
file_path = flags.data_base_dir
df_path = os.path.join(file_path,
                       '{}/speedDate{}{}{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
df = pd.read_csv(df_path)
data = df.values
oracle = pd.read_csv(
    file_path + '{}/speedDate{}{}Oracle{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
ITE_oracle = oracle['ITE'].values.reshape(-1, 1)

reader = pd.read_csv(file_path + flags.mod + '/ate_truth.csv')
truth = {
    'low': reader['low_ate'].values,
    'med': reader['med_ate'].values,
    'high': reader['high_ate'].values
}

Y = data[:, 0].reshape(-1, 1)
T = data[:, 1].reshape(-1, 1)
X = data[:, 2:]
index = X[:, flags.env].argsort()
X = np.delete(X, flags.env, axis=1)

share = int(index.shape[0] / 3)
env1 = index[:share]
env2 = index[share:share * 2]
env3 = index[share * 2: share * 3]


# Build environments
def make_environment(X, Y, T, ITE_oracle, env, e):
    if flags.collider == 1:
        X = X[env]
        for i in range(flags.num_col):
            np.random.seed(i)
            col_1 = Y[env] + T[env] + np.random.normal(0, e, len(env)).reshape(-1, 1)
            X[:, -(i + 1)] = np.squeeze(col_1)
    else:
        X = X[env]
#for runing with gpu & cpu
    if flags.gpu == 1:
        return {
            'covariates': torch.from_numpy(X).cuda(),
            'outcome': torch.from_numpy(Y[env]).cuda(),
            'treatment': torch.from_numpy(T[env]).cuda(),
            'ITE_oracle': torch.from_numpy(ITE_oracle[env])
        }
    else:
        return {
            'covariates': torch.from_numpy(X),
            'outcome': torch.from_numpy(Y[env]),
            'treatment': torch.from_numpy(T[env]),
            'ITE_oracle': torch.from_numpy(ITE_oracle[env])
        }


envs = [
    make_environment(X, Y, T, ITE_oracle,  env1, .01),
    make_environment(X, Y, T, ITE_oracle,  env2, .2),
    make_environment(X, Y, T, ITE_oracle, env3, 1),
]


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
        if flags.net =='tarnet':
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


mlp = MLP(X.shape[1])
if flags.gpu==1:
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
    if flags.gpu ==1:
        scale=torch.tensor(1.).cuda().requires_grad_()
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


pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc', 'train_ate', 'test_ate', 'train_t_acc',
             'test_t_acc')


def train_stack(val, env, all_train=True):
    if all_train == 1:
        return torch.stack([env[0][val], env[1][val], env[2][val]]).mean()
    else:
        return torch.stack([env[0][val], env[1][val]]).mean()


#different choice of optimizer yields different performance.
optimizer_adam = optim.Adam(mlp.parameters(), lr=flags.lr)
optimizer_sgd = optim.SGD(mlp.parameters(), lr=1e-7, momentum=0.9)

# training loop
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

    all_train = flags.all_train
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

    if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight
    if step < 501:
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
    #train with sgd after the performance is stablized with adam.
    else:
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()

    test_acc = envs[2]['acc']
    test_t_acc = envs[2]['t_acc']
    test_ate = envs[2]['ite'].mean()

    pred_ite = torch.stack([envs[0]['ite'], envs[1]['ite'], envs[2]['ite']]).detach().cpu().numpy()
    true_ite = torch.stack([envs[0]['ITE_oracle'], envs[1]['ITE_oracle'], envs[2]['ITE_oracle']]).detach().cpu().numpy()

    if step % 100 == 0:
        pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy() + train_t_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy(),
            np.mean(train_ate.detach().cpu().numpy()),
            np.mean(test_ate.detach().cpu().numpy()),
            train_t_acc.detach().cpu().numpy(),
            test_t_acc.detach().cpu().numpy()
        )
final_time = time.time()

# converting the tensor to numpy arrays
final_train_accs.append(train_acc.detach().cpu().numpy())
final_test_accs.append(test_acc.detach().cpu().numpy())
final_train_treatment_acc.append(train_t_acc.detach().cpu().numpy())
final_test_treatment_acc.append(test_t_acc.detach().cpu().numpy())

final_train_ate.append(train_ate.detach().cpu().numpy())
final_test_ate.append(test_ate.detach().cpu().numpy())

# print('Final train acc (mean/std across restarts so far):')
# print(np.mean(final_train_accs), np.std(final_train_accs))
#
# print('Final test acc (mean/std across restarts so far):')
# print(np.mean(final_test_accs), np.std(final_test_accs))
#
# print('Final train ate mae is:')
# print(abs(np.mean(final_train_ate) - truth[flags.dimension]))
#
# print('Final test ate mae is:')
# print(abs(np.mean(final_test_ate) - truth[flags.dimension]))
#
# print("PEHE is {}, std is: {}".format(np.square(pred_ite - true_ite).mean(), abs(pred_ite - true_ite).std()))
# print("MAE for ate is ", (abs((pred_ite).mean() - truth[flags.dimension])))

saver = {
    'pred_ite': [pred_ite],
    'sample_ite': [true_ite],
    'Y': torch.stack([envs[0]['outcome'], envs[1]['outcome'], envs[2]['outcome']]).detach().cpu().numpy(),
    'T': torch.stack([envs[0]['treatment'], envs[1]['treatment'], envs[2]['treatment']]).detach().cpu().numpy(),
    'index': [index]
}
if flags.net=='tarnet':
    tmp = os.path.join(flags.output_base_dir, 'tarnet/')
else:
    tmp = os.path.join(flags.output_base_dir, 'dragon/')

if flags.collider==1:
    tmp = os.path.join(tmp, 'collider/')
else:
    tmp = os.path.join(tmp, 'no_collider/')
if flags.all_train == 1:
    log_path = os.path.join(tmp, "all_train/")
else:
    log_path = os.path.join(tmp, "train_test/")

os.makedirs(log_path, exist_ok=True)

save_path = os.path.join(log_path, "{}/{}/".format(flags.mod, flags.dimension))
os.makedirs(save_path, exist_ok=True)

if flags.penalty_weight > 0:
    for num, output in enumerate([saver]):
        np.savez_compressed(os.path.join(save_path, "irm_ite_output_{}".format(str(flags.dat))), **output)
else:
    for num, output in enumerate([saver]):
        np.savez_compressed(os.path.join(save_path, "erm_ite_output_{}".format(str(flags.dat))), **output)

#
final_output = pd.DataFrame({
    'train_acc': final_train_accs,
    'test_acc': final_test_accs,
    'train_ate': final_train_ate,
    'test_ate': final_test_ate,
    'train_treatment_acc': final_train_treatment_acc,
    'test_treatment_acc': final_test_treatment_acc,
    'PEHE': np.square(pred_ite - true_ite).mean()
})


if flags.penalty_weight > 0:
    tmp = save_path+ 'irm_ate_output_{}.csv'.format(str(flags.dat))
else:
    tmp = save_path + 'erm_ate_output_{}.csv'.format( str(flags.dat))

final_output.to_csv(tmp, index=False)