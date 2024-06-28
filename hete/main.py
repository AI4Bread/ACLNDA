import sys
sys.path.append("..")
sys.path.append("../code")
from hete import utils
import seaborn as sb
import torch
from hete.model import ACLNDA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from layers import LogReg
from torch.utils.data import Dataset
import torch.nn.functional as F
import statistics
import argparse
import random


class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    



# Training
def train(g, feats, lr, epoch, device, lambda_loss, hidden_dim, sample_size=10,moving_average_decay=0.0,labels=None):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    in_dim = feats.shape[1]
    GNNModel = ACLNDA(in_dim, hidden_dim, 2, sample_size, lambda_loss=lambda_loss, tau=args.tau,  moving_average_decay=moving_average_decay, num_MLP=args.num_MLP)
    GNNModel.to(device)
    opt = torch.optim.Adam([{'params': GNNModel.parameters()}], lr=lr, weight_decay=0.0003)
    for i in tqdm(range(epoch)):
        feats = feats.to(device)
        loss, node_embeddings = GNNModel(g, feats, neighbor_dict, device=device)
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
        GNNModel.update_moving_average()
    return node_embeddings.cpu().detach(), loss.item()


def evaluate(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()

def write_results(acc, best_epoch):
    best_epoch = [str(tmp) for tmp in best_epoch]
    f = open("results/" + args.dataset + '_heterSSL', 'a+')
    f.write(args.dataset + ' --epochs ' + str(args.epoch_num) + ' --seed ' + str(args.seed) + ' --lr ' + str(args.lr) + ' --lambda_loss ' + str(args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(args.dimension) + ' --sample_size ' + str(args.sample_size) + ' --wd2 ' + str(args.wd2) + ' --num_MLP ' + str(args.num_MLP) + ' --tau ' + str(args.tau) + ' --best_epochs ' + " ".join(best_epoch) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f}\n')
    f.close()

def train_new_datasets(dataset_str, epoch_num = 10, lr = 5e-6, lambda_loss=1, sample_size=10, hidden_dim=None,moving_average_decay=0.0):
    g, labels, split_lists = utils.read_real_datasets(dataset_str)
    g = g.to(device)
    node_features = g.ndata['attr']
    node_labels = labels
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    acc = []
    epochs = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    #pre-training
    node_embeddings, loss = train(g, node_features, lr=lr, epoch=epoch_num, device=device,
                                  lambda_loss=lambda_loss, sample_size=sample_size, hidden_dim=hidden_dim,
                                  moving_average_decay=moving_average_decay)
    node_embeddings=node_embeddings.numpy()
    #np.savetxt("../dataset1_result/low_A_512_ACL_data1.txt", node_embeddings)
    #np.savetxt("../dataset1_result/low_A_512_ACL_data2.txt", node_embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="rna")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--lambda_loss', type=float, default=1)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=512)
    parser.add_argument('--moving_average_decay', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--wd2', type=float, default=1e-2)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2014)
    parser.add_argument('--gpu', type=int, default=1, help='GPU index.')

    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    dataset_str = args.dataset
    train_new_datasets(dataset_str=dataset_str, lr=args.lr, epoch_num=args.epoch_num, lambda_loss=args.lambda_loss, sample_size=args.sample_size, hidden_dim=args.dimension,moving_average_decay=args.moving_average_decay)
