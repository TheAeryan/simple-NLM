[![PyPI version](https://badge.fury.io/py/neural-logic-machine.svg)](https://badge.fury.io/py/neural-logic-machine)

# Simple NLM
A simple Pytorch implementation of Neural Logic Machines (arxiv.org/abs/1904.11694).

# How to use
Simply download from [PyPI](https://pypi.org/project/neural-logic-machine/) by doing `pip install neural-logic-machine`.
An alternative option is to clone the repository.

Assuming it has been installed from PyPI, using it is as simply as importing the `neural-logic-machine` package in Python:

```
from neural_logic_machine import NLM
import torch

# We create an NLM with two layers. The first layer receives different predicates
# and outputs 8 predicates for arities 0-3. The second layer receives these predicates and outputs 2 nullary predicates and one unary predicate.
# We do not need to specify the input size of the NLM (this is automatically inferred from the first forward pass).
nlm = NLM(hidden_features=[[8,8,8,8]], out_features=[2,1,0,0])

# Create a random input for the NLM
# The batch will contain two samples, the first one with 3 objects and the second one with 5 objects
# Note that we are not masking out those tensor positions corresponding to invalid objects (e.g., object 4 for the first batch element,
# despite it only having 3 objects). The reason is that there's no need to, as the NLM does that internally when computing the reduce operations
num_objs = [3,5]
input_NLM = [torch.randn((2,) + (5,)*r + (1,)) for r in range(3)] + [None] # The last position is None, as there are no ternary predicates
# Shapes of tensors in input_NLM: [(2,1), (2,5,1), (2,5,5,1), None]

# Forward pass
output = nlm(input_NLM, num_objs)
print(output) # Note in the result how invalid positions (those correspondings to objects 3 and 4 for the first batch element) are masked to -inf

>>> [tensor([[-0.0104,  0.4631],
        [-0.4774,  0.4594]], grad_fn=<AddmmBackward0>), tensor([[[-0.0508],
         [-0.0756],
         [-0.0551],
         [   -inf],
         [   -inf]],

        [[ 0.0538],
         [ 0.1004],
         [ 0.1422],
         [-0.0173],
         [ 0.0232]]], grad_fn=<MaskedFillBackward0>), None, None]
```

For training, we recommend using [Pytorch Lightning](https://lightning.ai/).

# Dependencies
Python 3 and Pytorch.

## Authors
- Carlos Núñez Molina
- Masataro Asai