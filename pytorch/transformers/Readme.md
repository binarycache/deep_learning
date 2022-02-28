Source:

http://peterbloem.nl/blog/transformers

https://jalammar.github.io/illustrated-transformer/

http://nlp.seas.harvard.edu/2018/04/03/attention.html


Paper:

Attention is all you need (https://arxiv.org/abs/1706.03762)

- We have an encoder decoder architecture.


#### Encoder Block ################################
1. Input (text in NLP, will go to the Embedding Layer).
    Input Shape: 
    Output Shape:
    
2. Positional Encoding will be added to the output of the embedding layer.
    Input Shape:
    Output Shape:
    
    
3.  The ouput from this embedding + encoding is sent to the Multi-Head Attention Layer:
    This layer expects input as key, query and value so the same input is copied and passed on as K,Q, and V.
    Input Shape:
    Output Shape:
    
4. Skip Connection from Step 2 + Normalization:

5. FeedForward:

6. Skip Connection from Step 4 + Normalization:

Step 3-6 is called a Transformer Block.

###### Decoder Block #####################################




###### Important Points #############################

1. Attention is the only operation in the whole architecture that propagates information between vectors. 
   Every other operation in the transformer is applied to each vector in the input sequence without interactions between vectors.
   
2. Self attention sees its input as a set, not a sequence. 
   If we permute the input sequence, the output sequence will be exactly the same, except permuted also (i.e. self-attention is permutation        equivariant). We will mitigate this somewhat when we build the full transformer, but the self-attention by itself actually ignores the          sequential nature of the input.
   
3. Thus, transformers are permutation equivariant i.e. the order of input does not matter. To remedy this **Positional Encoding** is used.
   
4. Basically, transformers can be thought of as a set-to-set transformations and thus they are much more general than any other DL architecture such as CNNs or RNNs.







###### FAQ ###############################

Q1: What is self-attention?
A1: output Yi = sum (WijXi)

Q2: What are query, keys and values?
A2: In the self-attention stage, every input vector is used in three ways.
    1. Compared to every other vector to establish the weights of its own output. (Query)
    2. Compared to every other vector to establish the weights for output of jth vector Yj. (Key)
    3. It is used as part of the weighted sum to compute each output vector once the weights are established. (Value)
    
    To get to these vectors or to make them play these roles, we introduce new parameters in the form of three matrices.
    
    
    