######################################################################
#Defining the contraction backend used for parallel processing.      #
#1. get_einsum_expr is changed to add 'Z' before all strings for     #
#   tensors to indicate that the 0th dimension (batch) is in parallel#
#2. torch.sum at the end of process_bucket has axis=1 instead of 0   #
#3. In get_sliced_bucked, transpose_order is modified to             #
#   tensor_transpose_order to be used in data.permute. We add 0 to   #
#   the beginning of the permutation order to preserve the batches   #
#4. In get_sliced_bucked, slice_bounds is initialized with           #
#   slice(None). This is the slice for the batch dimension, telling  #
#   it to keep all the batches.                                      #
######################################################################

from .qtensor.qtree.utils import num_to_alpha
import numpy as np
import gc

def get_einsum_expr(idx1, idx2):
    """
    Takes two tuples of indices and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)))
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}

    str1 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return 'Z' + str1 + ',' + 'Z' + str2 + '->' + 'Z' + str3

from .qtensor.contraction_backends import ContractionBackend
import torch
from .qtensor import qtree

class ParallelTorchBackend(ContractionBackend):

    def __init__(self, device = "cpu"):
        self.device = device
        self.cuda_available = torch.cuda.is_available()

    def process_bucket(self, bucket, no_sum=False):
        
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        
        for tensor in bucket[1:]:
            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )
            result_data = torch.einsum(expr, result_data, tensor.data)
            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )

        if len(result_indices) > 0:
            if not no_sum:  # trim first index
                first_index, *result_indices = result_indices
            else:
                first_index, *_ = result_indices
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        if no_sum:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=torch.sum(result_data, axis=1))
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                transpose_order = np.argsort(list(map(int, tensor.indices)))
                tensor_transpose_order = [0] + list(map(lambda x : x + 1, transpose_order))
                
                data = data_dict[tensor.data_key]
                #if self.cuda_available:
                #    cuda = torch.device('cuda')
                #    data = data.to(cuda)
                
                data = data.permute(tuple(tensor_transpose_order))
                # transpose indices
                indices_sorted = [tensor.indices[pp]
                                  for pp in transpose_order]

                # slice data
                slice_bounds = [slice(None)]
                for idx in indices_sorted:
                    try:
                        slice_bounds.append(slice_dict[idx])
                    except KeyError:
                        slice_bounds.append(slice(None))
                
                data = data[tuple(slice_bounds)]

                # update indices
                indices_sliced = [idx.copy(size=size) for idx, size in
                                  zip(indices_sorted, data.shape)]
                indices_sliced = [i for sl, i in zip(slice_bounds, indices_sliced) if not isinstance(sl, int)]
                assert len(data.shape) - 1 == len(indices_sliced)

                sliced_bucket.append(
                    tensor.copy(indices=indices_sliced, data=data))

            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return result.data