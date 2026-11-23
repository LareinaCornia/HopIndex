import numpy as np
import os
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def save_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(index, path)

def load_index(path):
    if os.path.exists(path):
        print(f"Successfully load trained index from {path}.")
        return load(path)
    else:
        print("No trained index locally.")
        return None

def euclidean_distance_func(X, Y):
    """
    Compute the Euclidean distance between two vectors X and Y.

    Parameters:
    X: the first vector
    Y: the second vector

    Return The squared Euclidean distance between the two vectors.
    """
    
    # TODO-[Warm up]: implement distance function

def do_query(query, vecs, k, distance_func = euclidean_distance_func):
    """
    Perform a query to find the k nearest vectors to the given query vector

    Parameters:
    query: The query vector, which is a 1D numpy array (128,)
    vecs: The database of vectors to search through
    k: The number of nearest vectors to return
    distance_func(optional): The distance function to use for measuring vector similarity (default is Euclidean distance)

    Returns:
    A tuple containing:
        sorted_distances: The distances to the k nearest vectors. For exmaple, [0.1, 0.2, 0.3]
        sorted_indices: The position of the nearest vectors in passing argument `vecs`. For example, [1, 3, 2]
    """

    # TODO-[Warm up]: implement query function
    D, I = (None, None)
    return D, I

class MyIndexFlat:
    def __init__(self):
        self._trained = False
        
        # index
        self.base = None

    def train(self, base):
        """
        Train the index. In flat index we just use base itself as index
        """
        self._trained = True
        self.base = base.copy()
    
    def search(self, query, k=5):
        """
        Parameters:
        query: The query vector, which is a 1D numpy array (128,)
        k: The number of nearest neighbors to return

        Returns:
        [D, I]:
            - D: The distances to the k nearest vectors
            - I: The indices of the k nearest vectors in real dataset
        """

        if not self._trained:
            raise ValueError("dataset should be trained first!")

        # TODO-[Part 1]: your code here
        D, I = (None, None)
        return D, I
    
    def insert(self, new_vecs):
        """
        Insert new vectors to index
        """
        if not self._trained:
            raise ValueError("dataset should be trained first!")

        # TODO-[Part 4]: your code here

class MyIndexIVF:
    def __init__(self):
        self._trained = False

        self.base = None
        # TODO-[Part 2]: add fields if you need
    
    def train(self, base, nclusters):
        """
        Train index. Clustering dataset

        Parameter:
        nclusters: the number of cluster partitioned
        """
        
        self._trained = True
        self.base = base.copy()

        # TODO-[Part 2]: your code here

    def search(self, query, k=1, nprobe=8):
        """
        Parameters:
        query: The query vector, which is a 1D numpy array (128,)
        k: The number of nearest neighbors to return
        nprobe: The number of nearest cluster to search

        Returns:
        [D, I]:
            - D: The distances to the k nearest vectors
            - I: The indices of the k nearest vectors in real dataset
        """
        if not self._trained:
            raise ValueError("dataset should be trained first!")

        # TODO-[Part 2]: your code here
        D, I = (None, None)
        return D, I
    
    def insert(self, new_vecs):
        """
        Insert new vectors to index
        """
        if not self._trained:
            raise ValueError("dataset should be trained first!")
        
        # TODO-[Part 2]: your code here

class MyIndexPQ:
    """
    Fields:
    m: The number of sub-vectors the original vector will be divided into
    d: The dimension of each sub-vector
    codes: Encoded vector (you shouldn't store original vector in IndexPQ)
    """
    def __init__(self, m):
        self.m = m
        self._trained = False

        self.d = None
        self.codes = None
        # TODO-[Part 3]: add fields if you need

    def _split_vector(self, vec):
        """
        Split vector to m sub-vectors

        Return:
        sub-vector: spiitted vector
        """
        # TODO-[Part 3]: implement _split_vector
        subvec = None
        return subvec

    def train(self, base, nbits):
        """
        Train index. Including generating encoded vectors

        Parameters:
        nbits: controls how many clusters each sub-vector will be partitioned into
        """

        self._trained = True
        # TODO-[Part 3]: your code here
    
    def search(self, query, k=1):
        """
        Parameters:
        query: The query vector, which is a 1D numpy array (128,)
        k: The number of nearest neighbors to return

        Returns:
        [D, I]:
            - D: The distances to the k nearest vectors
            - I: The indices of the k nearest vectors in real dataset
        """
        if not self._trained:
            raise ValueError("dataset should be trained first!")

        # TODO-[Part 4]: your code here
        D, I = (None, None)
        return D, I
    
    def insert(self, new_vecs):
        """
        Insert new vectors to index
        """
        if not self._trained:
            raise ValueError("dataset should be trained first!")

        # TODO-[Part 4]: your code here