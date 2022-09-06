import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
import numpy 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep
from torch_geometric.utils import dropout_adj


# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=16, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol, 256)
        self.mol_fc_g2 = torch.nn.Linear(256, 7 * output_dim)

        self.pro_conv1 = SAGEConv(num_features_pro, num_features_pro)
        self.pro_conv2 = SAGEConv(num_features_pro, num_features_pro * 2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 2, 256)
        self.pro_fc_g2 = torch.nn.Linear(256, 9 * output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(16 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        alpha = 0.05
        degree = 4
        emb1 = alpha * mol_x
        row1 = mol_edge_index[0]
        row1 = row1.cpu().numpy()
        col1 = mol_edge_index[1]
        col1 = col1.cpu().numpy()
        c1 = max(max(row1), max(col1))
        dim_1 = [1 for index in range(len(col1))]
        adj1 = coo_matrix((dim_1, (row1, col1)), shape=(c1 + 1, c1 + 1)).toarray()
        numpy.array(adj1, dtype=bool)
        adj1 = torch.from_numpy(adj1)
        adj1 = torch.as_tensor(adj1,dtype=torch.float32)
        for i in range(degree):
                mol_x = torch.spmm(adj1, mol_x)
                emb1 = emb1 + (1 - alpha) * mol_x / degree
        emb1 = torch.as_tensor(emb1, dtype=torch.float32)
        x = emb1
        x = gep(x, mol_batch)
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
