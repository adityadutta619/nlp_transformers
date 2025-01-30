## Overview
Transformers consist of an encoder and a decoder. First focusing on the encoder:

Within the encoder we will have a stack of self-attention layers. Attention is a mechanism by which each token in the input text can get different weights. So, each attention layer  Following is the process from the start. We get a sequence of tokens. Steps:
1. Split the given sentence into tokens using the tokenizer. 
2. Encode the tokens. 
3. These encodings pass through an embedding layer.
4. Post the embedding layer they pass into the multi-head self-attention layers - *Check this*

The encoder is composed of stack of encoder layers or blocks analogous to stacking convolutional layers in computer vision. 

### Short intro to convolutional neural networks
As a short intro to CNNS, Here each convolutional layer will have the following:
1. Kernel size, number of kernels. Kernels are analogous to filters
2. Stride
3. Padding
4. Activation function
The above when applied on an input image produces an output of size new_height = 1 + (height - K + 2P)/S, with channels=num_kernels. 

The total learnable parameters in the above case per kernel will be: (3 x 3 x 3) + 1 = 28
Total in the convolutional layer taking into consideration all kernels will be: 28 x num_filters 

