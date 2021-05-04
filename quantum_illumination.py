import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import time
import random


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.feature_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5 * 2),
                                       nn.ReLU()
                                       )
        self.feature_1 = nn.Sequential(nn.Linear(in_features=5 * 2, out_features=5 * 4),
                                       nn.ReLU(),
                                       nn.Linear(in_features=5 * 4, out_features=5 * 2),
                                       )
        self.act_1 = nn.ReLU()
        self.feature_2 = nn.Sequential(nn.Linear(in_features=5 * 2, out_features=5 * 4),
                                       nn.ReLU(),
                                       nn.Linear(in_features=5 * 4, out_features=5 * 2),
                                       )
        self.act_2 = nn.ReLU()
        self.feature_3 = nn.Sequential(nn.Linear(in_features=5 * 2, out_features=5 * 4),
                                       nn.ReLU(),
                                       nn.Linear(in_features=5 * 4, out_features=5 * 2),
                                       )
        self.act_3 = nn.ReLU()

        self.feature = nn.Sequential(
            nn.Linear(in_features=5 * 2, out_features=4),
            nn.Sigmoid()
        )
        m_1 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        m_2 = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1]])
        m_1 = torch.FloatTensor(m_1).unsqueeze(dim=0).unsqueeze(dim=1)
        m_2 = torch.FloatTensor(m_2).unsqueeze(dim=0).unsqueeze(dim=1)

        self.register_buffer('m_1', m_1)
        self.register_buffer('m_2', m_2)

    def forward(self, ENV_matrix, gamma):
        ENV = ENV_matrix[:, 0:1, 0:1, 0:1]
        x = torch.cat([ENV, gamma], dim=3)

        x = self.feature_0(x)

        x = x + self.feature_1(x)
        x = self.act_1(x)

        x = x + self.feature_2(x)
        x = self.act_2(x)

        x = x + self.feature_3(x)
        x = self.act_3(x)

        x = self.feature(x)
        x = x / (torch.sum(x, dim=3) + 1e-6)
        alpha = torch.sqrt(x)
        illu = torch.matmul(alpha.permute(0, 1, 3, 2), alpha)
        B_1 = torch.matmul(torch.matmul(self.m_1, illu), self.m_1.permute(0, 1, 3, 2))
        B_2 = torch.matmul(torch.matmul(self.m_2, illu), self.m_2.permute(0, 1, 3, 2))
        B = B_1 + B_2
        chan_0 = self.kron(ENV_matrix, B)
        chan_1 = gamma * illu + (1 - gamma) * chan_0
        return chan_0, chan_1, alpha

    def kron(self, a, b):
        """
        Kronecker product of matrices a and b with leading batch dimensions.
        Batch dimensions are broadcast. The number of them mush
        :type a: torch.Tensor
        :type b: torch.Tensor
        :rtype: torch.Tensor
        """
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)


class data_loader(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        return data

    def __len__(self):
        return len(self.dataset)


# Helstrom loss
def loss_fn1(chan_0, chan_1):
    chan = (0.5 * chan_0 - 0.5 * chan_1).squeeze(dim=0).squeeze(dim=0)
    nuc_loss = torch.norm(chan, p='nuc')
    loss = 0.5 * (1 - nuc_loss)

    return loss


# SWAP loss
def loss_fn2(chan_0, chan_1):
    chan = torch.matmul(chan_0, chan_1).squeeze(dim=0).squeeze(dim=0)
    tr = torch.trace(chan)
    loss = 0.5 * (1 + tr)

    return loss


def get_data(da, i):
    data = {}
    gamma = np.array(da[i][0])
    ENV = da[i][1]
    ENV_matrix = np.zeros([2, 2])
    ENV_matrix[0][0] = ENV
    ENV_matrix[1][1] = 1 - ENV
    gamma = torch.from_numpy(gamma).unsqueeze(dim=0).unsqueeze(dim=1)
    ENV_matrix = torch.from_numpy(ENV_matrix)
    data['ENV_matrix'] = ENV_matrix.unsqueeze(dim=0)
    data['gamma'] = gamma.unsqueeze(dim=0)
    return data

def adjust_learning_rate(optimizer, epoch, init_lr=0.01, min_lr=1e-6):
    """Sets the learning rate to the initial LR decayed by 2 every 5 epochs"""
    lr = init_lr * (0.5 ** (epoch // 5))
    if lr < min_lr:
        lr = min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # random seed
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # load data
    traindataset = []
    testdataset = []

    da_train = np.loadtxt('train.txt')
    for i in range(np.size(da_train, 0)):
        traindataset.append(get_data(da_train, i))
    da_test = np.loadtxt('test.txt')
    for j in range(np.size(da_test, 0)):
        testdataset.append(get_data(da_test, j))

    TrainLoader = data.DataLoader(data_loader(dataset=traindataset),
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0)
    TestLoader = data.DataLoader(data_loader(dataset=testdataset),
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)
    # training setting
    model = Generator()
    init_lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=init_lr) #, momentum=0.9)
    for epoch in range(50):
        # test
        for batch_idx, data in enumerate(TestLoader):
            model.eval()
            with torch.no_grad():
                gamma = data['gamma']
                ENV_matrix = data['ENV_matrix']
                chan_0, chan_1, alpha = model(ENV_matrix.float(), gamma.float())
                loss = loss_fn1(chan_0, chan_1)
            print(loss.item())

        time.sleep(0.05)
        # train
        adjust_learning_rate(optimizer, epoch, init_lr=init_lr)
        total_loss = 0
        tbar = tqdm(TrainLoader)
        for batch_idx, data in enumerate(tbar):
            model.train()
            gamma = data['gamma']
            ENV_matrix = data['ENV_matrix']
            chan_0, chan_1, alpha = model(ENV_matrix.float(), gamma.float())
            loss = loss_fn1(chan_0, chan_1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            desc = 'Training  : Epoch %d, Avg. Loss = %.5f' % (epoch+1, avg_loss)
            tbar.set_description(desc)
            tbar.update()