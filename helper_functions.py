# library imports
import random
import signal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from copy import deepcopy
from scipy.linalg import circulant
from itertools import product
from time import time
from func_timeout import func_timeout

np.set_printoptions(legacy = '1.25')

def Random_index(n):
    '''returns a random index as a tuple (i, j) in a n x n grid.
    '''
    return tuple([np.random.randint(n) for _ in range(2)])

def Neighbors(index, n):
    '''returns the neighbours of a given index in a n x n grid.
    '''
    i, j = index
    # compute the neighbours of an index, which are indices on 
    # top, bottom, right and left
    neighbors = np.array([(i + 1, j), 
                          (i - 1, j), 
                          (i, j + 1), 
                          (i, j - 1)])
    
    # if the index is in the edge, we need to remove the neighbors that spill out of the grid
    neighbors = neighbors[[indicator[0] & indicator[1] 
                            for indicator in (neighbors >= 0) & (neighbors < n) ]]
    
    return [tuple(map(int, neighbor)) for neighbor in neighbors]

def Is_partial_latin_square(U):
    '''Check if the matrix U is a partial latin square. 
    Let U be n x n. Then U has integers entries in {0, 1, ..., n}. The 0s
    represent empty cells. The function returns True if the filled cells c
    can be completed to form a latin square. 
    '''

    n = U.shape[0]
    
    # principle: every row and column must have unique non-zero values, 
    # that is, the number of unique non-zero values must equal the number
    # of non-zero values.

    '''To achieve this we neeed a slightly modified count function
    Why? In an unfinished board, for any row or column u, 
        #nonzero(u) = #unique(u) - 1, 
    as 0 will also be a unique entry. But when the board is complete we will have 
        #nonzero(u) = #unique(u). 
    These two cases are captured in the updated count function below. 
    '''
    Count_nonzero = lambda x: min(np.count_nonzero(x) + 1, n)

    for i in range(n):
        # row check 
        if len(np.unique(U[i, :])) != Count_nonzero(U[i, :]): return False
        # column check
        elif len(np.unique(U[:, i])) != Count_nonzero(U[:, i]): return False
        
    return True


def Is_valid(U, C):
    '''Checks if the current state of the puzzle is valid.
    '''
    # check if at the current state the inequalities ar satisfied
    for index, neighbor in C:
        if U[index] != 0 and U[neighbor] != 0:
        # can compare only if both indices are filled
            if U[index] > U[neighbor]:
                return False

    # once the inequalities are satisfied, 
    # check if the puzzle is a partial latin square
    return Is_partial_latin_square(U)
    
def Is_complete(U):
    '''checks if the grid U is complete, that is it has no 0s.
    '''
    return np.count_nonzero(U) == U.size

def Vanilla_backtrack(U, C):
    '''Solves the puzzle U using the backtracking (vanilla) algorithm.
    '''
    n = U.shape[0]

    if not Is_valid(U, C): return False # puzzle cannot be solved with the current orientation
    if Is_complete(U): return U # puzzle is solved

    # find the next empty index
    index = np.unravel_index(np.argmin(U, axis = None), U.shape)

    for i in range(1, n + 1):
        # iterate through candidate values to fill in the empty index
        U[index] = i
        Vanilla_backtrack.calls += 1 # count how many guess insertions are made
        if Vanilla_backtrack(U, C) is not False: return Vanilla_backtrack(U, C)

    U[index] = 0 # backtracking step (the solver has not returned yet, or a solution 
    # has not beeen found). The the index is reset to be tacked by another call.
    return False

def Display_grid(U, C = None):
    '''Given a grid U and constraints C, this method returns the puzzle in 
    a readable string format.
    '''
    # printing the grid without the inequalities
    if C is None:
        return "\n" + pd.DataFrame(U).to_string(index = False, header = False) + "\n"

    n = U.shape[0]

    # make a grid of size (2n - 1) x (2n - 1) to make room for the inequalities
    Grid = np.array([[''] * (2*n - 1)] * (2*n - 1), dtype = "U" + str(2*n)) 

    # replace the unfilled cells marked as 0s with "*"s.
    U = U.astype(str); U[U == '0'] = '*'

    # fill the grid with entry from U
    Grid[::2, ::2] = U
    # fill the grid with the inequalities
    for index, neighbor in C:
        i, j = index
        p, q = neighbor

        if i == p: # row inequalities here
            Grid[2*i, min(j, q) * 2 + 1] = '<' if j < q else '>'
        
        elif j == q: # column inequalities here
            Grid[min(i, p) * 2 + 1, 2*j] = 'A' if i < p else 'V'

    # use DataFrame to format the grid to have correct alignment
    return "\n" + pd.DataFrame(Grid).to_string(index = False, header = False) + "\n"

# # _____________________________________________ GRAPH FUNCTION BELOW

def Find_maximal_chains(edges):
    ''' Given a directed graph defined by edges, find all non-trivial maximal chains. That is, 
    the graph induced by the edges should be separable into "maximal" chains, where maximal means 
    that the chain is not contained in any other chain. 
    '''
    if not edges: return []
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Find all simple paths from source to sink nodes
    sources = [n for n, d in G.in_degree() if d == 0]
    sinks = [n for n, d in G.out_degree() if d == 0]
    
    all_chains = []
    for source in sources:
        for sink in sinks:
            # Find all simple paths between source and sink
            paths = nx.all_simple_paths(G, source, sink)
            for path in paths:
                if len(path) > 1:  # Only include chains with length > 1
                    all_chains.append(path)
    
    # Filter for maximal chains
    maximal_chains = []
    for chain in all_chains:
        is_maximal = True
        # simple n^2 comparison to check if chain is contained in any other chain
        # ?? Avenue for optimization 
        for other_chain in all_chains:
            if chain != other_chain and len(chain) < len(other_chain):
                # Check if chain is a subchain of other_chain
                chain_str = ''.join(map(str, chain))
                other_str = ''.join(map(str, other_chain))

                if chain_str in other_str:
                    is_maximal = False
                    break
                
        if is_maximal:
            maximal_chains.append(chain)
    
    return maximal_chains