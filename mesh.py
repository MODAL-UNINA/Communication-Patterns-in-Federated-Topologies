import psutil
import time
import random
import copy
import argparse
import sys
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle


class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = list(range(len(self.dataset[0])))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


class CNN_Mnist(nn.Module):
    def __init__(self, args):
        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# data split
_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
                    (len(sys.argv) == 1 and sys.argv[0] == '')

if _interactive_mode:
    from tqdm.auto import tqdm, trange
else:
    from tqdm import tqdm, trange

def is_interactive():
    return _interactive_mode

def myshow(plot_show=True):
    if _interactive_mode and plot_show:
        plt.show()
    else:
        plt.close()


# function for non-iid data split
def separate_data(data, num_clients, num_classes, least_samples=None, partition=None, alpha=0.1):
    X = {}
    y = {}
    statistic = {}

    dataset_content, dataset_label = data

    dataidx_map = {}

    if partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        statistic.setdefault(client, [])
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs].copy()
        y[client] = dataset_label[idxs].copy()

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
        
    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    print("Separate data finished!\n")

    ylabels = np.arange(num_clients)
    xlabels = np.arange(num_classes)

    x_mesh, y_mesh = np.meshgrid(np.arange(num_classes), np.arange(num_clients))

    s = np.zeros((num_clients,num_classes), dtype=np.uint16)
    for k_stat, v_stat in statistic.items():
        for elem in v_stat:
            s[k_stat, elem[0]] = elem[1]

    c = np.ones((num_clients,num_classes), dtype=np.uint16)
    R = s/s.max()/3

    viridis_cmap = cm.get_cmap('viridis')

    new_start = 0.5
    new_end = 1.0

    cm_color = cm.colors.LinearSegmentedColormap.from_list(
        'viridis_cut', viridis_cmap(np.linspace(new_start, new_end, 256))
    )

    fig, ax = plt.subplots(figsize=(0.4*num_clients, 0.4*num_classes))
    circles = [plt.Circle((i,j), radius=r) for r, j, i in zip(R.flat, x_mesh.flat, y_mesh.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cm_color, zorder=10)
    ax.add_collection(col)
    ax.set(title='Number of samples per class per client')
    ax.set(xlabel='Client ID', ylabel='Classes')
    ax.set(xticks=np.arange(num_clients), yticks=np.arange(num_classes),
        xticklabels=ylabels, yticklabels=xlabels)
    ax.set_xticks(np.arange(num_clients+1)-0.5, minor=True)
    ax.set_yticks(np.arange(num_classes+1)-0.5, minor=True)
    # plt.xticks(rotation=70)
    ax.grid(which='minor', zorder=0, alpha=0.5, color='white')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_facecolor("#E8EAED")
    myshow()

    return X, y, statistic


def test_img(net_g, datatest, args):
    net_g.eval()
    correct = 0

    all_pred = []
    all_target = []

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        all_pred.append(y_pred)
        all_target.append(target)

    accuracy = 100.00 * correct / len(data_loader.dataset)

    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_target = torch.cat(all_target, dim=0).cpu().numpy()

    f1 = f1_score(all_target, all_pred, average='macro') * 100
    precision = precision_score(all_target, all_pred, average='macro') * 100
    recall = recall_score(all_target, all_pred, average='macro') * 100

    return accuracy, precision, recall, f1



def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    global_count = sum([len(clients_trn_data[client_name][0]) for client_name in client_names])
    local_count = len(clients_trn_data[client_name][0])
    return local_count/global_count


def avg(w, weight_scalling_factor_list):
    w_global = copy.deepcopy(w[0])
    for i in w_global.keys():
        w_global[i] *= weight_scalling_factor_list[0]
    
    for k in w_global.keys():
        for i in range(1, len(w)):
            w_global[k] += weight_scalling_factor_list[i] * w[i][k]
    return w_global


# define args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
parser.add_argument('--model', type=str, default='cnn_mnist', help="model name")
parser.add_argument('--fl_method', type=str, default='mesh', help="name of federated learning method")

parser.add_argument('--no_clients', default=30, help="number of clients: K")
parser.add_argument('--num_clients', default=30, help="number of aggregation")
parser.add_argument('--partition', type=str, default='dir', help="split method of data, optional:'pat', 'dir'")
parser.add_argument('--alpha', type=float, default=0.5, help="parameter of dirichlet distribution")
parser.add_argument('--least_samples', type=int, default=100, help="the least samples of each client")

parser.add_argument('--rounds', type=int, default=200, help="rounds of communication")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
parser.add_argument('--bs', type=int, default=128, help="test batch size")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--seed', type=int, default=0, help="random seed (default: 1)")

args = parser.parse_args("")

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# dataset
if args.dataset == 'mnist':
    # use rotate on MNIST
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transforms.ToTensor())


    x_train = list(dataset_train.train_data)
    y_train = list(dataset_train.train_labels)

    dataset_content = np.stack(x_train, axis=0)
    dataset_label = np.stack(y_train, axis=0)


else:
    raise ValueError('Dataset Error')


X, y, statistic = separate_data((dataset_content, dataset_label), args.no_clients, args.num_classes, least_samples=args.least_samples, partition=args.partition, alpha=args.alpha)


clients_total = {}
for i in range(args.no_clients):
    imgs = torch.stack([transforms.ToTensor()(img) for img in X[i]])
    labels = y[i]
    clients_total[i] = (imgs, labels)



w_locals = {}
# ininialize the client model
for i in range(args.no_clients):
    local = LocalUpdate(args, dataset=clients_total[i])
    local_net = CNN_Mnist(args).to(args.device)
    w, _ = local.train(net=local_net)
    w_locals[i] = copy.deepcopy(w)


# training
acc_test_list = []
precision_test_list = []
recall_test_list = []
f1_test_list = []
data_transfer_volume_list = []


for iter in range(args.rounds):
    acc_test_per_round = []
    precision_test_per_round = []
    recall_test_per_round = []
    f1_test_per_round = []

    data_transfer_volume_per_round = []


    for i in range(args.no_clients):
        # all other clients, except Ci
        idxs = np.random.choice([j for j in range(args.no_clients) if j != i], args.num_clients-1, replace=False)

        # use the selected clients to update Ci
        new_w = avg([w_locals[i]] + [w_locals[j] for j in idxs], [1/args.num_clients] * args.num_clients)

        # train this model use Ci's dataset
        local = LocalUpdate(args, dataset=clients_total[i])
        # load weight
        net = CNN_Mnist(args).to(args.device)
        net.load_state_dict(new_w)
        w, energy_client = local.train(net=net)
        # update the weight of Ci
        w_locals[i] = copy.deepcopy(w)

        # record the Data Transfer Volume
        data_transfer_volume = 0
        for j in idxs:
            data_transfer_volume += sum([sys.getsizeof(w_locals[j] for j in idxs)])
        data_transfer_volume_per_round.append(data_transfer_volume)

    data_transfer_volume_per_round_sum = sum(data_transfer_volume_per_round)
    data_transfer_volume_list.append(data_transfer_volume_per_round_sum)

    # test all clients' model
    for i in range(args.no_clients):
        net = CNN_Mnist(args).to(args.device)
        net.load_state_dict(w_locals[i])
        acc, precision, recall, f1 = test_img(net, dataset_test, args)

        acc_test_per_round.append(acc)
        precision_test_per_round.append(precision)
        recall_test_per_round.append(recall)
        f1_test_per_round.append(f1)

    # get the average accuracy and loss
    acc_test = sum(acc_test_per_round) / len(acc_test_per_round)
    precision_test = sum(precision_test_per_round) / len(precision_test_per_round)
    recall_test = sum(recall_test_per_round) / len(recall_test_per_round)
    f1_test = sum(f1_test_per_round) / len(f1_test_per_round)


    if acc_test >= 60:
        print('Round {:3d}, Average Accuracy: {:.2f}'.format(iter, acc_test))

    acc_test_list.append(acc_test)
    precision_test_list.append(precision_test)
    recall_test_list.append(recall_test)
    f1_test_list.append(f1_test)


# save the result
result = {
    'acc_test': acc_test_list,
    'precision_test': precision_test_list,
    'recall_test': recall_test_list,
    'f1_test': f1_test_list,
    'data_transfer_volume': data_transfer_volume_list,
}

with open('result_mesh_{}_{}.pkl'.format(args.no_clients, args.num_clients), 'wb') as f:
    pickle.dump(result, f)
