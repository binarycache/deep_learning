import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.module):
    '''
    Implementation of a simple multi-head attention
    
    Note: This is not the optimized version, here we will have n_heads copies of 
    query, key and value.
    
    '''
    
    def __init__(self, k:int, n_heads:int = 8):
        '''
        Args:
        ---------------------------
        k: Embedding dimension 
        n_heads: Number of multi-attention heads
        
        Notes:
        ---------------------------
        We think of the h attention heads as h separate sets of three matrices 
        𝐖_q^r, 𝐖_k^r,𝐖_v^r, but it's actually more efficient to combine these for 
        all heads into three single k×hk matrices, 
        so that we can compute all the concatenated queries, keys and values 
        in a single matrix multiplication.
        
        '''
        super().__init__()
        self.k = k 
        self.n_heads = n_heads
        # These compute the queries, keys and values for all 
        # heads (as a single concatenated vector)
        # this will generate the weight matrices Wq^r, Wk^r, Wv^r
        self.tokeys    = nn.Linear(k, k * n_heads, bias=False)
        self.toqueries = nn.Linear(k, k * n_heads, bias=False)
        self.tovalues  = nn.Linear(k, k * n_heads, bias=False)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unifyheads = nn.Linear(n_heads * k, k)
        
        def forward(self, x):
            '''
            Shape of x: batch_size, t, k where t is the no. of vectors 
            and k is the size of each vector
            i.e. x contains all the word embedding vectors
            '''
            b, t, k = x.size()
            h = self.n_heads
            # view is similar to numpy's reshape but it doesn't create any copies
            # The output of each of the linear layers will be 
            # b, t, h*k which we are reshaping as (b,t,h,k)
            queries = self.toqueries(x).view(b, t, h, k)
            keys    = self.tokeys(x).view(b, t, h, k)
            values  = self.tovalues(x) .view(b, t, h, k)
            
            # Next, we need to compute the dot products. This is the same operation
            # for every head, so we fold the heads into the batch dimension. 
            # This ensures that we can use torch.bmm() as before, 
            # and the whole collection of keys, queries and values 
            # will just be seen as a slightly larger batch.
            # Since the head and batch dimension are not next to each other,
            # we need to transpose before we reshape. 
            # (This is costly, but it seems to be unavoidable.)
            # contiguous means to make a copy of data so that the order of elements 
            # remains unchanged inspite of using view.
            queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
            keys = queries.transpose(1,2).contiguous().view(b*h,t,k)
            values = queries.transpose(1,2).contiguous().view(b*h,t,k)
            
            # let's scale these before doing dot product.
            # instead scale the keys and queries by fourth root of k before multiplying them together. 
            # This should save memory for longer sequences.
            queries = queries / (k ** (1/4))
            keys    = keys / (k ** (1/4))
            
            
            # - get dot product of scaled queries and keys
            dot = torch.bmm(queries, keys.transpose(1, 2))
            # - dot has size (b*h, t, t) containing raw weights

            dot = F.softmax(dot, dim=2) 
            # - dot now contains row-wise normalized weights
            
            # apply the self attention to the values
            out = torch.bmm(dot, values).view(b, h, t, k)
            
            # swap h, t back, unify heads
            out = out.transpose(1, 2).contiguous().view(b, t, h * k)
            return self.unifyheads(out)