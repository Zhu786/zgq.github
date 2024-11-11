import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import numpy as np
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc15 = GraphConvolution(nhid, nhid*2)
        # self.gc16 = GraphConvolution(nhid*2, nhid*4)
        self.gc2 = GraphConvolution(nhid*2, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc15(x, adj))
        # x = F.dropout(x, self.dropout, training = self.training)
        # x = F.relu(self.gc16(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid,nhid*4)
        self.gc2 = GraphConvolution(nhid*4, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc15(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc16(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False, device='cuda'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj.to_dense())
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class UMGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout,sadj,sadjj,sadjjj,fadj,fadjj,fadjjj):
        super(UMGCN, self).__init__()
        self.estimated_sadj = nn.Parameter(torch.FloatTensor(n, n))
        self.estimated_sadj1 = nn.Parameter(torch.FloatTensor(n, n))

        self.estimated_fadj = nn.Parameter(torch.FloatTensor(n, n))
        self.estimated_fadj1 = nn.Parameter(torch.FloatTensor(n, n))

        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self.estimated_adj1 = nn.Parameter(torch.FloatTensor(n, n))

        self._init_estimation(((sadj + sadjj + sadjjj) / 3).to_dense())
        self._init_estimation1(((fadj + fadjj + fadjjj) / 3).to_dense())
        self._init_estimation2((((sadj + fadj) / 2 + (sadjj + fadjj) / 2 + (sadjjj + fadjjj) / 2) / 3).to_dense())


        self.normalized_sadj = self.normalize()
        self.normalized_fadj = self.normalize1()
        self.normalized_adj = self.normalize2()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN3 = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN3 = GCN(nfeat, nhid1, nhid2, dropout)

        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.symmetric = False
        self.nclass = nclass
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.w_weight = 0
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )
        self.m1 = nn.Linear(nhid2, nclass)
        self.m2 = nn.LogSoftmax(dim=1)
        self.cluster_layer = Parameter(torch.Tensor(nclass, nhid2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def _init_estimation(self, sadj):
        with torch.no_grad():
            n = len(sadj)
            self.estimated_sadj.data.copy_(sadj)

    def _init_estimation1(self, fadj):
        with torch.no_grad():
            n = len(fadj)
            self.estimated_fadj.data.copy_(fadj)

    def _init_estimation2(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_sadj, self.estimated_fadj

    def normalize(self):
        sadj = self.estimated_sadj
        normalized_sadj = self._normalize(sadj + torch.eye(sadj.shape[0]))
        return normalized_sadj

    def normalize1(self):
        fadj = self.estimated_fadj
        normalized_fadj = self._normalize(fadj + torch.eye(fadj.shape[0]))
        return normalized_fadj

    def normalize2(self):
        adj = self.estimated_adj
        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    def forward(self, x, sadj, sadjj,sadjjj,fadj,fadjj,fadjjj):

        self.estimated_sadj1 = Parameter(self.normalized_sadj.cuda())
        self.estimated_fadj1 = Parameter(self.normalized_fadj.cuda())
        self.estimated_adj1 = Parameter(self.normalized_adj.cuda())


        emb1 = self.SGCN1(x, sadj)

        com1 = self.CGCN(x, self.estimated_sadj1)
        com2 = self.CGCN1(x, self.estimated_fadj1)
        com3 = self.CGCN2(x, self.estimated_adj1)

        emb2 = self.SGCN2(x, sadjj)
        emb3 = self.SGCN3(x,sadjjj)
        femb1 = self.FGCN1(x, fadj)
        femb2 = self.FGCN2(x, fadjj)
        femb3 = self.FGCN3(x, fadjjj)


        semb = torch.stack([emb1, emb2, emb3], dim=1)
        femb = torch.stack([femb1, femb2, femb3], dim=1)
        semb, satt = self.attention(semb)
        femb, fatt = self.attention(femb)
        semb1 = semb * 0.5 + com1
        femb1 = femb * 0.5 + com2
        cemb = (semb + femb) * 0.5 + com3
        emb = (semb1 + femb1 + cemb) / 2

        output1 = self.m1(emb)
        output = self.m2(output1)

        pslb = output

        cluster_assignment = torch.argmax(pslb, -1)

        onesl=torch.ones(pslb.shape[0]).cpu()
        zerosl=torch.zeros(pslb.shape[0]).cpu()
        weight_label=0
        # print(weight_label)
        cluster_assignment1= F.one_hot(cluster_assignment,self.nclass)
        label = torch.topk(cluster_assignment1, 1)[1].squeeze(1)
        print('Predition Label:',label)
        # print('label',label)
        # output = self.MLP(emb)
        # print(output1.unsqueeze(1).shape,'output1.unsqueeze(1)')
        # print(self.cluster_layer.shape,'output1.unsqueeze(1)')
        s = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        s = s.pow((1 + 1.0) / 2.0)
        s = (s.t() / torch.sum(s, 1)).t()
        return output, s, emb1, com1, com2, emb2,emb3,semb,femb,femb1,femb2,femb3,self.estimated_sadj,self.estimated_fadj,self.estimated_adj,output1,label,weight_label
