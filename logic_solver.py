# library imports
import random
import signal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from copy import deepcopy
from scipy.linalg import circulant
from itertools import product
from time import time
from func_timeout import func_timeout

from helper_functions import *

np.set_printoptions(legacy = '1.25')

# # __________________________________ Logic solver helper functions

def A2S(a):
    '''Converts an array to a comma separated string.
    '''
    return ','.join(map(str, a))

def S2A(s):
    '''Converts a comma separated string to an array.
    '''
    if s == '': return [] # empty string
    if ',' not in s: return [int(s)] # singleton 
    return list(map(int, s.split(',')))    
    
def Remove(s, val):
    '''Removes val from the string s. If val is in s it additionally returns -1, 
    if not it returns 0. (this function is a helper function to interact with 
    the Count matrix in Logic_solver)
    '''
    if val in S2A(s):
        return A2S(list(set(S2A(s)) - {val})), -1
    
    return s, 0

# # __________________________________  Logic solver strategies

'''Solving strategies
1. Elimination 
    a) Inequality_elimination
    b) Line_elimination

2. Insertion
    a) Inequality_insertion
    b) Line_insertion

3. Backtracking
'''

def Pivot(Count):
    '''Given the Count array, it returns the location of the "pivot". A 
    pivot is a SOLVED index, that is Count[pivot] = 1. If pivot doesn't exist, 
    the function returns False.  
    '''
    index = np.unravel_index(np.argmin(Count), Count.shape)

    if Count[index] == 1: 
        return index # pivot is found

    if Count[index] > 1: 
        return None # the grid has no pivot
    
    if Count[index] < 1: 
        return False # grid is unsolvable

def Insert(R, Count, C, insertions):
    '''Given the grid info (R, Count, C) and a list of insertinos in the (index, value) format.
    This method will insert the values and update the grid, following by performing
    customary elimination and insertion steps.
    '''

    n = R.shape[0]

    # IMP: solved every index inserted by this method will have Count[index] = n + 1
    # this will be an important distinction between solved indices for which the grid has been
    # reduced to other solved indices (where Count[index] = 1) where reduction is pending
    
    for index, val in insertions:
        R[index] = str(val)
        Count[index] = n + 1 # solved index is inserted
        R, Count = Line_elimination(R, Count, C, pivot = index) 
        # Line_elimination has to be done for every insertion separately

    # conduct other reductions
    R, Count = Line_insertion(R, Count, C)
    R, Count = Inequality_insertion(R, Count, C)

    # the above reduction may have created new solved indices which need to inserted
    R, Count = Insert_all_pivots(R, Count, C)
    
    # IMP: once solved indices are inserted, they do not 
    # play a role in the future reductions.

    return R, Count

def Insert_all_pivots(R, Count, C):
    '''
    This function reduces the grid by continuously inserting solved indices
    until no more pivots can be found.
    '''
    pivot = Pivot(Count)
    while pivot:
        R, Count = Insert(R, Count, C, [(pivot, int(R[pivot]))])
        pivot = Pivot(Count)

    return R, Count

def Inequality_elimination(R, Count, C):
    '''Given the residue grid R, we first identify the chains formed by the inequalities. 
    This inlcludes the trivial chains of length 1, i.e. the inequalities themselves. 

    Now being a member of a chain renders some extremal values impossible. 
    For example, in the chain * < * < * < *, the last entry has to be larger than 4, and
    the first entry cannot be n. Such values are removed, and the updated grid is returned.

    This method only needs to be applied to the initial grid. 
    '''
    n = R.shape[0]

    # find all maximal chains (this is done by constructing a graph using the inequalities 
    # as the edge set and then surveying all the paths in the graph). 
    # Note, this step can be costly if len(C) is large. 
    chains = Find_maximal_chains(C)

    # for each chain eliminate the extremals
    for chain in chains:
        c = len(chain)
        for i in range(c):
            index = chain[i]

            # important step: the rule of exclusion is below, as the following must be satisfied
            # R[index] >= i + 1 and R[index] <= (n + 1) - (c - i) 
            for val in S2A(R[index]):
                if val < i + 1:
                    # eliminate exclusions enforced by '>'
                    # (this entry has to be larger than so many values)
                    R[index], count = Remove(R[index], val)
                    Count[index] += count

                if val > (n + 1) - (c - i):
                    # eliminate exclusions enforced by '<'
                    R[index], count = Remove(R[index], val)
                    Count[index] += count
                    
    return R, Count

def Line_elimination(R, Count, C, pivot):
    '''The method recursively reduces the rows and columns of a logic grid. The 
    solved position (pivot) is given, and this method removes the solved value
    from all the other entries in the row and column of the pivot.
    '''
    # stopping criteria
    if np.min(Count) < 1:
        # the grid is unsolvable. 
        return R, Count

    n = R.shape[0]

    i, j = pivot

    for l in range(n):
        # row reduce every entry except the pivot
        if l != j:
            R[i, l], count = Remove(R[i, l], int(R[i, j]))
            Count[i, l] += count
        # column reduce every entry except the pivot
        if l != i:
            R[l, j], count = Remove(R[l, j], int(R[i, j]))
            Count[l, j] += count
    
    return R, Count

def Line_insertion(R, Count, C):
    '''For every row and column collect the residues of all the unsolved indices. 
    If there is any value that appears only once in the logic, insert it in the 
    appropriate position. 
    '''

    n = R.shape[0]

    for i in range(n): # for ROWS
        if np.min(Count) < 1: return R, Count # grid is not solvable

        solved_values = [int(r) for r in R[i, : ] if ',' not in r]

        collection = [r for r in R[i, : ] if ',' in r]
        if len(collection) == 0: # all the indices are solved
            continue 
        
        # freq is a frequency array of the unsolved residues in the current row. 
        # The 0th entry is masked because 0 doesn't appear in the residue. 
        freq = np.bincount(S2A(','.join(collection)))
        freq[0] = n + 1 # remove from minimum computation

    
        for val in range(len(freq)):
            if freq[val] == 1 and val not in solved_values: # value is unique
                # and value is not already solved
                
                for j in range(n): # find the column index of val in residue R 
                    if val in S2A(R[i, j]): # index found
                        R, Count = Insert(R, Count, C, [((i, j), val)])
                        break

    for j in range(n): # for COLUMNS
        if np.min(Count) < 1: return R, Count # grid is not solvable
  
        solved_values = [int(r) for r in R[ : , j] if ',' not in r]
        
        # collect the residues of all the unsolved indices
        collection = [r for r in R[ : , j] if ',' in r]
        if len(collection) == 0: # all the indices are solved
            continue 

        # freq is a frequency array of the unsolved residues in the current column.
        # The 0th entry is masked because 0 doesn't appear in the residue.
        freq = np.bincount(S2A(','.join(collection)))
        freq[0] = n + 1 # remove from minimum computation
        
        for val in range(len(freq)):
            if freq[val] == 1 and val not in solved_values: # value is unique
                # and value is not already solved
                
                for i in range(n): # find the row index of val in residue R
                    if val in S2A(R[i, j]): # index found
                        R, Count = Insert(R, Count, C, [((i, j), val)])
                        # here is a prospective error of unwanted looping
                        # when REI is called from insert, it will get stuck here again
                        break

    return R, Count

def Inequality_insertion(R, Count, C):
    ''' This method goes through every constraint and forms satisfiability 
    pairs based on the current state of the residue grid. If the pair is
    unique, the method enforces an insertion.
    '''
    # stopping criteria
    if np.min(Count) < 1:
        # some index has no possible values
        return R, Count

    n = R.shape[0]

    for index, neighbor in C:
        # list of all pairs below
        all_pairs = list(product(S2A(R[index]), S2A(R[neighbor])))

        if len(all_pairs) == 1:
            if all_pairs[0][0] >= all_pairs[0][1]:
                # the inequality is satisfied
                R[index], R[neighbor] = '', ''
                Count[index], Count[neighbor] = 0, 0
                return R, Count

        if len(all_pairs) > 1: # ensuring that index and neighbor are not both solved
            # compute all valid pairs
            valid_pairs = [pair for pair in all_pairs if pair[0] < pair[1]]
            if len(valid_pairs) == 1:
                # insert when there is only one valid pair
                R, Count = Insert(R, Count, C, [(index, valid_pairs[0][0]),
                                                (neighbor, valid_pairs[0][1])])

    return R, Count

def Backtrack_on_residue(R, Count, C, randomize = True):
    ''' Solve the puzzle using a backtracking algorithm on the residue grid.'''
    
    n = R.shape[0]

    # stopping criteria
    if np.min(Count) < 1:
        # some index has no possible values
        return False, R, Count
    
    if np.min(Count) == n + 1:
        # all the indices are solved
        return True, R, Count
    
    # find an unsolved index with the smallest count (minimum possible values)
    index = np.unravel_index(np.argmin(Count), Count.shape)

    _R, _Count = deepcopy(R), deepcopy(Count)

    # possible values
    values = S2A(R[index])

    if randomize: np.random.shuffle(values)

    for val in values:
        # iterate through candidate values to fill in the empty index
        R, Count = Insert(R, Count, C, [(index, val)])
        Backtrack_on_residue.calls += 1 # count how many guess insertions are made

        # recursive call
        Bool, R, Count = Backtrack_on_residue(R, Count, C, randomize = randomize)
        if Bool: # a solution has been found
            return Bool, R, Count
        
        # the current val didn't lead to a solution. 
        # Try next val, but on the original residue grid
        R, Count = deepcopy(_R), deepcopy(_Count)   
        
    # backtracking step (the solver has not returned yet, or a solution
    # has not been found). The residue is reset to be tacked by another call.
    return False, _R, _Count

# # __________________________________ Logic solver main function

def Logic_solver(U, C, randomize = True):
    '''The logic solver will undergo the following checks to narrow 
    down the solution space. Once all the information is extracted, 
    a backtracking algorithm will run.
    '''

    n = U.shape[0]

    # RESIDUE GRID SETUP

    # R for "residue" will hold a list of all possible values available at each index 
    R = np.array([[A2S(list(range(1, n + 1)))] * n] * n)
    Count = np.array([[n] * n] * n)  # count of possible values at each index

    # update the revealed indices in the grid in the residue grid
    R[U != 0] = U[U != 0].astype(str)
    Count[U != 0] = 1 # solved indices will have Count = 1

    # # ___________________ ENFORCED CHECKS (insetions & eliminations)

    # 1. Inequality elimination
    R, Count = Inequality_elimination(R, Count, C)
    
    # 2. Line insertion
    R, Count = Line_insertion(R, Count, C)
    
    # 3. Inequality insertion
    R, Count = Inequality_insertion(R, Count, C)

    # 4. Line elimination only works when there are pivots
    R, Count = Insert_all_pivots(R, Count, C)

    if Pivot(Count) is False:
         # after reductions the residue at some index has become empty.
         # therefore the grid is unsolvable
         return False

    # # _____________________ IMP: This is the furthest the Residue grid 
    # can be reduced! Beyond this point we will try to insert values 
    # and see if we reach a solution, using a backtracking algorithm.

    # Backtrack
    Bool, R, Count = Backtrack_on_residue(R, Count, C, randomize = randomize)
    
    # return solved grid by converting it to int, or False if the grid is unsolvable
    return R.astype(int) if Bool else False 