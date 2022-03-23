"""
Operations to load/contract quantum circuits. All functions
operating on Buckets (without any specific framework) should
go here.
"""

import itertools
import random

random.seed(0)


class Var(object):
    """
    Index class. Primarily used to store variable id:size pairs
    """
    def __init__(self, identity, size=2, name=None):
        """
        Initialize the variable
        identity: int
              Index identifier. We use mainly integers here to
              make it play nicely with graphical models.
        size: int, optional
              Size of the index. Default 2
        name: str, optional
              Optional name tag. Defaults to "v[{identity}]"
        """
        self._identity = identity
        self._size = size
        if name is None:
            name = f"v_{identity}"
        self._name = name
        self.__hash = hash((identity, name, size))

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def identity(self):
        return self._identity

    def copy(self, identity=None, size=None, name=None):
        if identity is None:
            identity = self._identity
        if size is None:
            size = self._size

        return Var(identity, size, name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.identity)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.identity == other.identity
                and self.size == other.size
                and self.name == other.name)

    def __lt__(self, other):  # this is required for sorting
        return self.identity < other.identity


class Tensor(object):
    """
    Placeholder tensor class. We use it to do manipulations of
    tensors kind of symbolically and to not move around numpy arrays
    """
    def __init__(self, name, indices,
                 data_key=None, data=None):
        """
        Initialize the tensor
        name: str,
              the name of the tensor. Used only for display/convenience.
              May be not unique.
        indices: tuple,
              Indices of the tensor
        shape: tuple,
              shape of a tensor
        data_key: int
              Key to find tensor's data in the global storage
        data: np.array
              Actual data of the tensor. Default None.
              Usually is not supplied at initialization.
        """
        self._name = name
        self._indices = tuple(indices)
        self._data_key = data_key
        self._data = data
        self._order_key = hash((self.data_key, self.name))

    @property
    def name(self):
        return self._name

    @property
    def indices(self):
        return self._indices

    @property
    def shape(self):
        return tuple(idx.size for idx in self._indices)

    @property
    def data_key(self):
        return self._data_key

    @property
    def data(self):
        return self._data

    def copy(self, name=None, indices=None, data_key=None, data=None):
        if name is None:
            name = self.name
        if indices is None:
            indices = self.indices
        if data_key is None:
            data_key = self.data_key
        if data is None:
            data = self.data
        return Tensor(name, indices, data_key, data)

    def __str__(self):
        return '{}({})'.format(self._name, ','.join(
            map(str, self.indices)))

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self._order_key < other._order_key

    def __mul__(self, other):
        if self._data is None:
            raise ValueError(f'No data assigned in tensor {self.name}')
        if self.indices == other.indices:
            return self.copy(data=self._data * other._data)
        elif len(self.indices) == 0 or len(other.indices) == 0:
            # Scalar multiplication
            return self.copy(data=self._data * other._data)
        else:
            raise ValueError(f'Index mismatch in __mul__: {self.indices} times {other.indices}')

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.name == other.name
                and self.indices == other.indices
                and self.data_key == other.data_key
                and self.data == other.data)



def bucket_elimination(buckets, process_bucket_fn,
                       n_var_nosum=0):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we eliminate
    buckets one by one.

    Parameters
    ----------
    buckets : list of lists
    process_bucket_fn : function
              function that will process this kind of buckets
    n_var_nosum : int, optional
              number of variables that have to be left in the
              result. Expected at the end of bucket list
    Returns
    -------
    result : numpy.array
    """
    # import pdb
    # pdb.set_trace()
    n_var_contract = len(buckets) - n_var_nosum

    result = None
    for n, bucket in enumerate(buckets[:n_var_contract]):
        if len(bucket) > 0:
            tensor = process_bucket_fn(bucket)
            if len(tensor.indices) > 0:
                # tensor is not scalar.
                # Move it to appropriate bucket
                first_index = int(tensor.indices[0])
                buckets[first_index].append(tensor)
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor

    # form a single list of the rest if any
    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        # only multiply tensors
        tensor = process_bucket_fn(rest, no_sum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result



def reorder_buckets(old_buckets, permutation):
    """
    Transforms bucket list according to the new order given by
    permutation. The variables are renamed and buckets are reordered
    to hold only gates acting on variables with strongly increasing
    index.

    Parameters
    ----------
    old_buckets : list of lists
          old buckets
    permutation : list
          permutation of variables

    Returns
    -------
    new_buckets : list of lists
          buckets reordered according to permutation
    label_dict : dict
          dictionary of new variable objects
          (as IDs of variables have been changed after reordering)
          in the form {old: new}
    """
    # import pdb
    # pdb.set_trace()
    if len(old_buckets) != len(permutation):
        raise ValueError('Wrong permutation: len(permutation)'
                         ' != len(buckets)')
    perm_dict = {}
    for n, idx in enumerate(permutation):
        if idx.name.startswith('v'):
            perm_dict[idx] = idx.copy(n)
        else:
            perm_dict[idx] = idx.copy(n, name=idx.name)

    n_variables = len(old_buckets)
    new_buckets = []
    for ii in range(n_variables):
        new_buckets.append([])

    for bucket in old_buckets:
        for tensor in bucket:
            new_indices = [perm_dict[idx] for idx in tensor.indices]
            bucket_idx = sorted(
                new_indices, key=int)[0].identity
            # we leave the variables permuted, as the permutation
            # information has to be preserved
            new_buckets[bucket_idx].append(
                tensor.copy(indices=new_indices)
            )

    return new_buckets, perm_dict