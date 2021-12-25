import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, SumAttention
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nsub): # nhid[64,16,16,256]
        super(GCN, self).__init__()
        self.nsub = nsub
        self.nhid1 = nhid[1]
        self.nhid2 = nhid[2]
        self.gc1 = GraphConvolution(nfeat, nhid[0])
        self.dropout = dropout
        self.gc2 = GraphConvolution(nhid[0], nhid[1]) # [N,36,n_node,nhid0->nhid1]

        self.sum = SumAttention(nhid[1])
        self.fc0 = nn.Linear(nhid[1],nhid[2]) #[N,nhid2] -> [N, nhid1]
        self.bn = nn.BatchNorm1d(nhid[2])
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(nsub*nhid[2], nhid[3])
        self.bn1 = nn.BatchNorm1d(nhid[3])
        self.relu1 = nn.ReLU()

        nhid4 = 64
        self.fc2 = nn.Linear(nhid[3], nhid4) # nhid4=56 && no fc2  pcc=0.7761
        self.bn2 = nn.BatchNorm1d(nhid4)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(nhid4, 1)

    def forward(self, x, adj):
        f = F.relu(self.gc1(x, adj))
        f = F.dropout(f, self.dropout, training=self.training)
        f = self.gc2(f, adj) # [batchsize,36,n_node,n_heat]
        f = torch.sum(f, axis=2).reshape([-1,self.nhid1])
        f = self.relu(self.bn(self.fc0(f))).reshape([-1,self.nsub*self.nhid2])
        f = self.relu1(self.bn1(self.fc1(f)))
        f = self.fc(self.relu2(self.bn2(self.fc2(f))))
        return f