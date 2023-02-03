import torch
from torch import Tensor, nn
import torch.nn.functional as F

from madress_2023.train.config import Config

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, config: Config, out_dim):
        super().__init__()
        
        self.config = config

        # Attention + mapping to output.
        self.linear1 = nn.Linear(config.dim_hidden, 2*config.dim_hidden)
        self.linear2 = nn.Linear(2*config.dim_hidden, 1)
        self.linear3 = nn.Linear(config.dim_hidden, out_dim)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: Tensor):

        # Attention weights over sequence.
        att = self.linear2(self.dropout(self.activation(self.linear1(x)))) 
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)

        # Pool sequence to vector using attention weights.
        x_pooled = torch.bmm(att, x).squeeze(1)

        # Map vector to output.
        out = self.linear3(x_pooled)
        
        return out



class Model(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.do_ad = config.do_ad
        self.do_mmse = config.do_mmse

        # Normalization.
        self.norm = nn.BatchNorm1d(config.dim_input)

        # Down-projection.
        self.down_proj = nn.Linear(
            in_features=config.dim_input,
            out_features=config.dim_hidden,
        )
        self.down_proj_drop = nn.Dropout(config.dropout)
        self.down_proj_act = nn.ReLU()

        # Attention pooling.
        assert self.do_ad or self.do_mmse
        if self.do_ad:
            self.pool_ad = PoolAttFF(config, 2)
        if self.do_mmse:
            self.pool_mmse = PoolAttFF(config, 1)
            self.sigmoid_msme = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        
        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))

        # Down-projection to hidden dim.
        x = self.down_proj(x)
        x = self.down_proj_act(x)
        x = self.down_proj_drop(x)

        ## AD
        if self.do_ad:
            out = self.pool_ad(x)
            return out
        
        ## MMSE
        if self.do_mmse:
            out = self.pool_mmse(x).squeeze(-1)
            out = self.sigmoid_msme(out)
            return out
