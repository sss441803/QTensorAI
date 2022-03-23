###########################################################################
#Changing the Var class in qtree.optimizer. The Var class gets initialized#
#to the wrong size when used in parallel mode. Here, size = 2 is forced   #
###########################################################################

class ParallelVar(object):
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
        size = 2
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

        return ParallelVar(identity, size, name)

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
                and self.name == other.name)

    def __lt__(self, other):  # this is required for sorting
        return self.identity < other.identity