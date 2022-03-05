import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.module):
    '''
    Implementation of a simple multi-head attention
    
    Note: This is not the optimized version, here we will have n_heads copies of 
    query, key and value.
    
    '''
    
    def __init__(self, k:int, n_heads:int = 8, mask=False):
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
        self.mask = mask
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
            
            
            # - get dot product of scaled queries and keys
            dot = torch.bmm(queries, keys.transpose(1, 2))
            
            # scaling the dot product
            dot = dot / math.sqrt(k) 
            
            # - dot has size (b*h, t, k) containing raw weights
            assert dot.size() == (b*h, t, k), f'Matrix has size {dot.size()}, expected {(b*h, t, k)}.'
            
            if self.mask: # mask out the lower half of the dot matrix,including the diagonal
                mask_(dot, maskval=float('-inf'), mask_diagonal=False)
            
            dot = F.softmax(dot, dim=2) 
            # - dot now contains row-wise normalized weights
            assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan
            
            if self.mask == 'first':
                dot = dot.clone()
                dot[:, :1, :] = 0.0
                # - The first row of the first attention matrix is entirely masked out, 
                # so the softmax operation results
                #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

            # apply the self attention to the values
            out = torch.bmm(dot, values).view(b, h, t, k)
            
            # swap h, t back, unify heads
            out = out.transpose(1, 2).contiguous().view(b, t, h * k)
            return self.unifyheads(out)