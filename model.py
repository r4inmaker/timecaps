# IMPORTS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import trange


# MODEL
class Cell_A(nn.Module):
    def __init__(self, cp, ap, g2, cSA, aSA, g3):
        super(Cell_A, self).__init__()
        assert g3 % 2 != 0, 'make me odd, father'
        
        self.cp,  self.ap  = cp,  ap
        self.cSA, self.aSA = cSA, aSA
        
        self.psi_m = nn.Conv1d(
            in_channels = self.cp * self.ap,
            out_channels = self.cp * self.ap,
            kernel_size = g2,
            stride = 1,
            padding = 'same',
            bias = False
        )
        
        # compute pad_A (g3 must be odd)
        pad_A = int( g3/2 - 0.5 )
        
        self.psi_m_A = nn.Conv2d(
            in_channels = 1,
            out_channels = self.cSA * self.aSA,
            kernel_size = (self.ap, g3),
            stride = (self.ap, 1),
            padding = (0, pad_A),
            bias = True
        )
        
    def forward(self, X):
        N, C, L = X.shape
        assert C == self.cp * self.ap, 'cp * ap == K'
        
        # Φ_A_Conv
        X = self.psi_m(X) 
        
        # Ω_A_PTC
        sq_norm = (X**2).sum(-1).unsqueeze(-1)
        X = ((sq_norm / (1 + sq_norm)) * X / sq_norm).unsqueeze(1)
        
        # W_A_Conv
        X = self.psi_m_A(X)
        
        return X
    
    
    
class Cell_B(nn.Module):
    def __init__(self, cb, ab, ins, outs, Ln, n):
        super(Cell_B, self).__init__()
        
        self.cb, self.ab = cb, ab
        self.Ln, self.n = Ln, n
        
        self.Conv_B = nn.Conv1d(
            in_channels = ins,
            out_channels = outs,
            kernel_size = 1
        )
    
    def forward(self, X):
        
        X = self.Conv_B(X).view(-1, self.cb * self.ab, self.n, self.Ln)
        
        return X
    
    
class Routing(nn.Module):
    def __init__(self, n_iters=3):
        super(Routing, self).__init__()
        
        self.n_iters = n_iters
        
    def forward(self, X):
        
        # a tryhard attempt at making this efficient (feat. einsum 🤓)
        N, C, n_caps, L = X.shape
        b = torch.zeros(N, C, n_caps, 1)
        
        for i in range(self.n_iters - 1):
            
            c = b.softmax(-2)
            
            sj = (c * X).sum(-2).unsqueeze(-2)
            
            # squash
            sq_norm = (sj**2).sum(-1).unsqueeze(-1)
            vj = (sq_norm / (sq_norm + 1)) * (sj / sq_norm)
            
            # update b
            bj = torch.einsum('abij,abjk->abik', X, vj.permute(0,1,3,2))
            b += bj
        
        return vj.squeeze(-1)
            
        
    
class TimeCaps(nn.Module):
    def __init__(self, L=450, K=64, g1=7, cp=8,   ap=8, g2=5,
                       n=10,              cSA=8,  aSA=8, g3=3,
                                          cb=8,   ab=4):
        super(TimeCaps, self).__init__()
        
        assert L % n == 0, 'try again G'
        

        self.K, self.g1 = K, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3
        self.cb, self.ab = cb, ab
        
        self.L = L
        self.Ln = L // n 
        self.n = n
        
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = K,
            kernel_size = g1,
            stride = 1,
            padding = 'same',
            bias = False
        )
        
        self.cell_A = Cell_A(
            cp=self.cp, ap=self.ap, g2=self.g2, cSA=self.cSA, aSA=self.aSA, g3=self.g3
        )
        
        self.cell_B = Cell_B(
            cb=self.cb, ab=self.ab, ins=K, outs=cb*ab, Ln=self.Ln, n=self.n
        )
        
        self.routing = Routing()
        
    def forward(self, X):
        
        # Φ_Conv1
        X = self.conv1(X)
    
        # Temporal Capsules [A]
        X_A = self.cell_A(X)
        
        # Routing [A]
        X_A = self.routing(X_A).reshape(-1, self.cSA, self.aSA * self.L)
        
        # Spacial Capsules [B]
        X_B = self.cell_B(X)
        
        return X_A


