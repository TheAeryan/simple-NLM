"""
Authors: Carlos Núñez Molina and Masataro Asai

Unpadded Neural Logic Machine (NLM) implementation.
This implementation sacrifices speed for memory savings.
It is especially recommended for batches where elements contains vastly different numbers of objects
(e.g., a batch with 1 sample with 10 objects and another sample with 100 objects).

An issue with the standard NLM implementation is that
it requires a large amount of zero-padding for handling different number of objects.

Imagine concatenating two binary NLM representations as follows.

  [B1, 10, 10, 7] and [B2, 20, 20, 7] -> [B1+B2, 20, 20, 7]

Notice that B1*10*10*3*7 floats are zero-filled.

This implementation takes advantage of the fact that the linear layer is a convolution over the last dimension.
In essense, it performs the following operation:
                                                              linear                  restore shape
  [B1, 10, 10, 7] and [B2, 20, 20, 7] -> [B1*10*10+B2*20*20, 7] -> [B1*10*10+B2*20*20, Q_2] -> [B1, 10, 10, Q_2] and [B2, 20, 20, Q_2]

This trick saves up memory by avoiding zero-padding. 
However, it is several times slower in GPU than the standard NLM implementation, due to the increased use of list comprehensions.
"""

from typing import Union, Literal, Optional
ResidualType = Literal[None, "input", "all"]
Activation = Literal["sigmoid", "relu"]

import torch
from torch import nn
import numpy as np
import itertools
import functools

class _InferenceMLP(nn.Module):
    def __init__(self, arity:int, hidden_size:int, output_size:int, batchnorm:bool, activation:Activation, activate_output:bool=True):
        super().__init__()

        self.arity = arity
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.activate_output = activate_output
        self.batchnorm = batchnorm

        activation_class = {'sigmoid':nn.Sigmoid, 'relu':nn.ReLU}[activation]

        if hidden_size < 1: # Do not use hidden layer
            if activate_output:
                self.layers = nn.Sequential(
                    nn.LazyLinear(output_size),
                    activation_class())
            else:
                self.layers = nn.Sequential(
                    nn.LazyLinear(output_size))
        else: # Use hidden layer
            if activate_output:
                self.layers = nn.Sequential(
                    nn.LazyLinear(hidden_size),
                    activation_class(),
                    nn.Linear(hidden_size, output_size),
                    activation_class())
            else:
                self.layers = nn.Sequential(
                    nn.LazyLinear(hidden_size),
                    activation_class(),
                    nn.Linear(hidden_size, output_size))

        # Apply batch normalization to the outputs
        if batchnorm:
            self.layers.append(nn.BatchNorm1d(output_size, momentum=0.005))

    def forward(self, Xs:list[torch.Tensor], num_obj_list:list[int]):
        # Let B the batch size.
        # Xs is a list of tensors. Lets denote their shape as:
        # Xs: [O_1, O_1, F], [O_2, O_2, F], .... [O_B, O_B, F].
        F = Xs[0].shape[-1]
        Xs_flatten_concat = torch.cat([X.view(-1, F) for X in Xs]) # [\sum_i O_i*O_i, F]
        mlp_output = self.layers(Xs_flatten_concat).flatten()      # [\sum_i O_i*O_i, Q]

        # < Recover the original (unflattened) shape >

        # Shapes of the output tensor for each batch element
        output_tensor_shapes = [(num_obj,)*self.arity + (self.output_size,) for num_obj in num_obj_list]
        # Number of elements in mlp_output corresponding to each batch element
        num_flatten_outputs_each_sample = [self.output_size*num_obj**self.arity for num_obj in num_obj_list]

        # Separate the mlp_output into a list of flattened tensors, where each tensor contains the mlp_output
        # for each sample in the batch
        mlp_outputs_flatten = torch.split(mlp_output, num_flatten_outputs_each_sample)

        # Unflatten the tensor of each sample
        mlp_outputs_unflatten = [tensor.view(tensor_shape) for tensor, tensor_shape in \
                                 zip(mlp_outputs_flatten, output_tensor_shapes)]
                   
        return mlp_outputs_unflatten


class _NLM_Layer(nn.Module):
    """
    Creates the NLM layer and initializes the inference MLPs. There is a different inference MLP
    to compute each output predicate arity. For example: if out_features = [0, 8, 4, 0] (only compute
    output unary and binary predicates), then there are two different inference MLPs for this layer.

    @in_features List with the number of input predicates of each arity, in ascending order. It is equal to
                                out_features for the previous layer.
                                Example: [8, 8, 4, 0] -> The previous NLM layer's output consisted of 8 nullary predicates,
                                                          8 unary predicates and 4 binary predicates.
    @out_features List with the number of output predicates of each arity, in ascending order.
    @mlp_hidden_features Units in the hidden layer of all the inference MLPs. If 0, the inference MLPs have no hidden layer.
    @activate_output Whether the MLPs should apply an activation function to the output of the last layer.
    @residual_connections If True, we add residual connections. This means we append, for each different arity, the input
                          predicates to the output predicates.
    @exclude_self If True, the NLM ignores tensor positions corresponding to repeated indexes (e.g., [5][5][3] or [2][2][0][1])
                  when performing the reduce operation
    @reduce_mask_cache A list 
                  
    Note: if we use residual_connections, @in_features must consider the extra predicates (due to the residual
          connections) but @out_features must NOT consider the extra predicates.
    Note 2: if we use residual_connections for the NLM, all NLM layers <except for the last one> must use residual_connections
    """
    def __init__(self, out_features:list[int], mlp_hidden_features:int, batchnorm, activation,
                 activate_output, residual_connections=True, exclude_self=True,
                 reduce_masks_cache=None):
        super().__init__()

        self.out_features = out_features
        self.residual_connections = residual_connections
        self.exclude_self = exclude_self
        max_arity = out_features.shape[0]-1
        self.reduce_masks_cache = reduce_masks_cache

        if exclude_self and max_arity > 3:
            raise NotImplementedError("Can't use exclude_self=True for NLMs with breadth > 3")

        if exclude_self and reduce_masks_cache is None:
            raise Exception("If exclude_self is True, then reduce_masks_cache must contain the initial reduce masks")

        self._mlps = nn.ModuleList([_InferenceMLP(i, mlp_hidden_features, out_features[i], batchnorm, activation, activate_output) \
                                   if out_features[i] > 0 else None \
                                   for i in range(len(out_features))])


    def _compute_mask(self, arity, num_objs, device):
        assert arity > 0, "We can't reduce nullary predicates"
        assert arity < 4, "No support for tensors of arity bigger than 3"

        # Unary predicates -> all the positions are valid (since there are no repeated objects)
        if arity == 1:
            return torch.ones(num_objs, device=device).unsqueeze(-1)
        # Binary predicates -> mask has 1 in every position except for the diagonal
        elif arity == 2:
            return (1 - torch.eye(num_objs, device=device)).unsqueeze(-1)
        # Ternary predicates -> mask has 1 in every position except where two variables are the same (x==y, x==z or y==z)
        else:
            binary_mask = 1 - torch.eye(num_objs, device=device)
            ternary_mask = functools.reduce(lambda a,b: a*b, [torch.unsqueeze(binary_mask, dim=dim) for dim in range(3)])
            return ternary_mask.unsqueeze(-1)

    def _append_reduce_masks_to_cache(self, max_objs, device):
        curr_num_objs = len(self.reduce_masks_cache[0][0])-1 # Max num objs for which reduce masks are currently saved in cache
        max_arity = len(self.reduce_masks_cache[0])
        
        with torch.no_grad():
            for num_objs in range(curr_num_objs+1, max_objs+1):
                for arity in range(1, max_arity+1):
                    self.reduce_masks_cache[0][arity-1].append(self._compute_mask(arity, num_objs, device)) # arity-1 because we do not save reduce masks for arity 0
                    self.reduce_masks_cache[1][arity-1].append(1-self.reduce_masks_cache[0][arity-1][num_objs]) # Also store 1-m

    def _get_mask(self, ind, arity, num_objs, device):
        """
        Obtains a mask for duplicate arguments needed by the reduce operation when exclude_self=True.
        If ind==0, we return the reduce masks for the "max" reduction and, if ind==1, we return the masks for the
        "min" reduction (corresponding to 1-m).
        In order to save up computation time, we save the computed masks in the cache "self.reduce_masks_cache".
        """
        # Retrieve the mask from the cache if it exists. If not, compute the masks up to num_objs and save them
        # in cache for later use.
        max_objs_in_cache = len(self.reduce_masks_cache[0][0])-1

        if num_objs > max_objs_in_cache:
            self._append_reduce_masks_to_cache(num_objs, device)

        return self.reduce_masks_cache[ind][arity-1][num_objs] # arity-1 because we do not save reduce masks for arity 0

    def _expand(self, X, num_obj_list):
        """
        We transform a tensor of predicates of arity a into predicate a+1, by adding an additional variable to each predicate
        and copying the predicate values along the new axis (corresponding to the new variable).

        Example (2 predicates of arity 1, instantiated on 3 objects): input shape = [3, 2] -> output shape = [3, 3, 2]

        @X A list containing the tensors for each element in the batch (i.e., list[i] contains the tensors for i-th batch element).
        @num_obj_list A list where each element contains the number of objects of the sample X[i].
        """
        num_dims = X[0].dim()
        num_objs = len(num_obj_list)
        repeat_tensors = torch.stack( [torch.ones(num_objs, dtype=torch.int) for _ in range(num_dims-1)] + [torch.tensor(num_obj_list, dtype=torch.int)] + [torch.ones(num_objs, dtype=torch.int)], dim=1).tolist()

        expanded_tensors = [tensor.unsqueeze(-2).repeat(repeat_tensor) for tensor, repeat_tensor in zip(X, repeat_tensors)]
        return expanded_tensors

    def _reduce(self, Xs, arity, reduce_type):
        """
        We transform a tensor of predicates of arity a into predicate a-1, by deleting the last variable by taking the maximum
        or minimum element along the corresponding axis.

        Example (2 predicates of arity 2, instantiated on 3 objects): input shape = [3, 3, 2] -> output shape = [3, 2]

        @Xs A list containing the tensors of the corresponding arity for each element in the batch (i.e., list[i] contains the tensors for i-th batch element).
        @reduce_type Either 'min' (corresponding to "forall") or 'max' (corresponding to "exists")
        """
        # If reduce_type=="min", we calculate the min along the -2 axis, else we take the maximum

        if not self.exclude_self:
            reduced_tensors = [torch.amin(X, -2) for X in Xs] if reduce_type == 'min' else \
                              [torch.amax(X, -2) for X in Xs]

            return reduced_tensors

        else:
            # Obtain the torch_device from the input data (CPU or GPU)
            data_device = Xs[0].device

            if reduce_type == 'max':
                reduced_tensors = [torch.amax(X * self._get_mask(0, arity, X.shape[0], data_device), -2) for X in Xs]

            elif reduce_type == 'min':
                reduced_tensors = [torch.amin(X * self._get_mask(0, arity, X.shape[0], data_device) + \
                                              self._get_mask(1, arity, X.shape[0], data_device), -2) for X in Xs]
                # torch.amin(X * m + (1-m), -2)
            else:
                raise NotImplementedError("Right now, only 'max' and 'min' reductions are implemented")

            return reduced_tensors

    def _permute(self, X):
        """
        We return a tensor with all the possible permutations of the input tensor's axes that index objects.
        The extra tensors are appended as if they were additional predicates.

        Example: input shape = [3, 3, 2] -> ouput_shape = [3, 3, 4]
        Example2: input shape = [3, 3, 3, 1] -> ouput_shape = [3, 3, 3, 6]

        <Note>: the new tensors corresponding to permutations are views of the original tensor, in order to reduce
          memory consumption. However, this means they share the reference!

        @X A list containing the tensors for each element in the batch (i.e., list[i] contains the tensors for i-th batch element).
        """
        tensor_dim = X[0].dim() # All the sample tensors in the batch have the same number of dimensions

        if tensor_dim < 3: # dim=1 -> nullary predicates, dim=2 -> unary predicates : they do not need permutations
            return X

        obj_axes = list(range(tensor_dim-1)) # Indexes of axes that correspond to objects in the tensor (e.g.: [0, 1] for binary predicates)
        obj_axes_permutations = list(itertools.permutations(obj_axes))
        last_axis = (tensor_dim-1,) # Axis corresponding to the predicates, which is NOT permuted

        # For each tensor in X, concatenate along the predicate dimension (torch.cat(..., dim=-1)) all the possible permutations
        # of the axes that index objects (obj_axes_permutations)
        permuted_tensors = [ torch.cat([tensor.permute(perm + last_axis) for perm in obj_axes_permutations], dim=-1) \
                             for tensor in X ]

        return permuted_tensors


    def forward(self, X:list[torch.Tensor], num_obj_list:list[int]) -> list[torch.Tensor]:
        """
        It receives a list @input_tensors_list with the input tensors (corresponding to the output tensors of the previous NLM layer)
        and returns a list with the output tensors of the current NLM layer.
        If there are no input tensors for arity r, then @input_tensors_list[r] must be None.

        @X A list with all the tensors for all the samples in the batch.
           X[r] is a list with the predicates of arity r for all the samples.
           X[r][i] corresponds to the predicates of arity r for the i-th sample.
        @num_obj_list A list where each element contains the number of objects of the sample X[r][i] for any arity r.
        """
        # A list with the predicates of all the arities but just for the first sample in the batch
        first_sample = [tensors[0] if tensors is not None else None for tensors in X]
        max_arity = len(first_sample)-1

        # Obtain the <real> input tensors for the MLP of each arity r
        # By <real> we mean we take into account the additional predicates from the expand, reduce and permute operations
        real_input_tensors_list = []
        first_sample_len = len(first_sample)

        for r in range(first_sample_len):
            # If we do not need the output predicates for arity r, we skip this arity
            if self.out_features[r] > 0:

                # Expand arity r-1
                if r > 0 and first_sample[r-1] is not None:
                    expanded_tensors = self._expand(X[r-1], num_obj_list)
                else:
                    expanded_tensors = None

                # Tensors arity r
                if first_sample[r] is not None:
                    curr_tensors = X[r]
                else:
                    curr_tensors = None

                # Reduce arity r+1, with min and max
                if r < max_arity and first_sample[r+1] is not None:
                    reduced_tensors_min = self._reduce(X[r+1], r+1, 'min')
                    reduced_tensors_max = self._reduce(X[r+1], r+1, 'max')
                else:
                    reduced_tensors_min = None
                    reduced_tensors_max = None

                # Concatenate the tensors

                # Iterate over all existing tensors
                if expanded_tensors is None:
                    if curr_tensors is None:
                        if reduced_tensors_min is None:
                            raise ValueError("fError: no real input tensors to compute output predicates of arity {r}")
                        else:
                            zip_iterator = zip(reduced_tensors_min, reduced_tensors_max)
                            concatenated_tensors = [torch.cat(x, dim=-1) for x in zip_iterator]# Concatenate the input tensors of each sample in the batch
                    else:
                        if reduced_tensors_min is None:
                            concatenated_tensors = curr_tensors
                        else:
                            zip_iterator = zip(curr_tensors, reduced_tensors_min, reduced_tensors_max)
                            concatenated_tensors = [torch.cat(x, dim=-1) for x in zip_iterator]
                else:
                    if curr_tensors is None:
                        if reduced_tensors_min is None:
                            concatenated_tensors = expanded_tensors
                        else:
                            zip_iterator = zip(expanded_tensors, reduced_tensors_min, reduced_tensors_max)
                            concatenated_tensors = [torch.cat(x, dim=-1) for x in zip_iterator]
                    else:
                        if reduced_tensors_min is None:
                            zip_iterator = zip(expanded_tensors, curr_tensors)
                            concatenated_tensors = [torch.cat(x, dim=-1) for x in zip_iterator]
                        else:
                            zip_iterator = zip(expanded_tensors, curr_tensors, reduced_tensors_min, reduced_tensors_max)
                            concatenated_tensors = [torch.cat(x, dim=-1) for x in zip_iterator]


                # Permute object axes (for arity < 2, self._permute() simply returns the tensor without permuting anything)
                permuted_tensors = self._permute(concatenated_tensors)

                # Append the final tensor to the list of real input tensors
                real_input_tensors_list.append(permuted_tensors)
            else:
                real_input_tensors_list.append(None) # Append None if we do not need the real input tensors for arity r


        # Obtain the output tensors by applying the MLP to the real input tensors
        # Also, if we are using residual_connections, append the input predicates to the output predicates arity-wise
        output_tensors_list = []
        out_features_len = len(self.out_features)

        num_samples_in_batch = len(X[0])

        for r in range(out_features_len):
            if self.out_features[r] == 0: # We do not need to compute the output predicates for this arity

                if self.residual_connections:
                    output_tensors_list.append(X[r])
                else:
                    output_tensors_list.append([None]*num_samples_in_batch)

            else:
                out_tensors = self._mlps[r](real_input_tensors_list[r], num_obj_list) # Obtain the output tensor using the MLP corresponding to arity r

                if self.residual_connections and X[r][0] is not None:
                    # Concatenate the output tensors to the input tensors
                    X_r = X[r]
                    out_tensors_cat = [torch.cat(x, dim=-1) for x in zip(X_r, out_tensors)]
                    output_tensors_list.append(out_tensors_cat)
                else:
                    output_tensors_list.append(out_tensors)

        return output_tensors_list

class NLM(nn.Module):
    """
    Main class. It implements an entire NLM composed of several layers.

    Constructor parameters:
        @hidden_features List that contains, for each layer, the number of output predicates for each arity, in ascending order.
                         Example: 2 hidden layers, 8 predicates of arities 0-3 -> [[8, 8, 8, 8], [8, 8, 8, 8]]
        @out_featurs List with the number of predicates of each arity for the output layer of the NLM
                     Example: if the NLM only predicts a single nullary predicate -> [1,0,0,0]
        @mlp_hidden_features Units in the hidden layer of all the inference MLPs. If 0, the inference MLPs have no hidden layer.    
        @residual Type of residual for the NLM. The options are:
                  - None: no residual connections
                  - "input": concatenate the NLM input as additional input to each NLM layer (except the first one)
                  - "all": concatenate the inputs of all the previous NLM layers as additional input to each NLM layer (except the first one)
        @exclude_self If True, the reduce operation ignores tensor positions corresponding to repeated indexes (e.g., [5][5][3] or [2][2][0][1])
        @batchnorm Whether to apply batch normalization to the output of the inference MLPs
        @activation Activation function for the inference MLPs. The options are: "sigmoid" and "relu"
    """
    def __init__(self,
                 hidden_features:list[list[int]],
                 out_features:list[int],
                 mlp_hidden_features:int = 0,
                 residual : ResidualType = "input",
                 exclude_self : bool = True,
                 batchnorm : bool = False,
                 activation : Activation = "sigmoid"):
        super().__init__()

        all_output_features = np.array(hidden_features+[out_features], dtype=int)
        depth, max_arity = all_output_features.shape[0], all_output_features.shape[1]-1

        self.residual = residual

        # Cache that contains the masks used by the _reduce() method when exclude_self=True
        # This cache is shared between all the _NLM_Layer objects in order to save up memory and computation time
        # reduce_masks_cache[0] stores the reduce masks, whereas reduce_masks_cache[1] stores the opposite, i.e.,
        # 1-reduce_masks_cache[0]
        # reduce_masks_cache[0][r][n] stores the reduce masks for the 'max' operation corresponding to arity=r and
        # num_objs=n
        reduce_masks_cache = [[[None] for _ in range(max_arity)] for _ in range(2)]

        self.layers = nn.ModuleList([_NLM_Layer(all_output_features[i],
                                                mlp_hidden_features,
                                                activation=activation,
                                                batchnorm            = (i != depth-1 and batchnorm),
                                                activate_output      = (i != depth-1),
                                                residual_connections = (i != depth-1 and residual == "all"),
                                                exclude_self = exclude_self,
                                                reduce_masks_cache = reduce_masks_cache) \
                                     for i in range(len(hidden_features)+1)])

        self.num_layers = len(self.layers)

    def forward(self, xs:list[Optional[list[torch.Tensor]]], num_objs:Union[int, list[int]]):
        """
        Compute a forward pass with the NLM.

        @xs A list with the tensors for all the samples in the batch. xs[r] contains the tensors of arity r for all the samples.
            xs[r][i] contains the tensors of arity r for the i-th sample in the batch. It has shape [num_objs[i]*r, P], where
            num_objs[i] contains the number of objects of the i-th sample and P is the number of input predicates for arity r. 
            If there are no predicates for arity r, then xs[r] must be None. 
        @num_objs A list with the number of objects for each sample in the batch. If instead of a list an integer is provided,
                  then @xs needs to correspond to a single-sample batch: xs = [tensor_r0, tensor_r1, ...]
        """

        # Detect whether @xs corresponds to a single sample or a batch of samples
        # If it corresponds to a single sample, we encode it as a single-sample batch
        if type(num_objs) == int: # @xs corresponds to a single sample
            xs = [[x] for x in xs]
            num_objs = [num_objs]

        # A = len(xs)             # max arity
        # B = len(xs[0])          # batch size

        def concatenate(*xs):
            return torch.cat([ x for x in xs if x is not None ], dim=-1)

        arities = range(len(xs))
        xss = [xs]
        for i in range(self.num_layers):
            if self.residual == "input" and i > 0: # concat input layer
                xss.append(
                    self.layers[i](
                        [ [concatenate(x,y) for x,y in zip(xss[-1][r], xss[0][r])]
                          for r in arities],
                        num_objs))
            elif self.residual == "all" and i > 0: # concat all previous layers
                xss.append(
                    self.layers[i](
                        [ [concatenate(xss_r) for xss_r in zip(*[xss[i][r] for i in range(len(xss))])]
                          for r in arities],
                        num_objs))
            else:
                xss.append(
                    self.layers[i](
                        xss[-1],
                        num_objs))

        return xss[-1]
