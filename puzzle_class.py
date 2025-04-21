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

from helper_functions import *
from logic_solver import *

np.set_printoptions(legacy = '1.25')

class Futoshiki_puzzle:
    ''' Class attributes are initialized in __init__ method.
    '''
    def __init__(self, n = 5, 
                       S = None,
                       C = None,
                       U = None):
        
        # Class attributes
        self.n = n # size of the puzzle
        self.S = S # solved puzzle matrix (will be a latin square)
        self.C = C # list of inequalities
        self.U = U # unsolved puzzle

    def Generate_puzzle(self, k = 3,
                              r = 1):  
        '''Generates a puzzle of size n x n with k constraints and r pre-assigned cells.
        '''

        if self.S is None:
            # generate a circulant and permute its rows and columns
            S = circulant(np.array(range(1, self.n + 1)))
            np.random.shuffle(S); np.random.shuffle(S.T)
            # are there latin squares that are not permutations of the circulant? YES.

            self.S = S

        # generate k random constraints
        self.C = []
        pairs = [] # keep a track of pairs seen already, to ensure unique pairs

        while True: # keep generating until we have a valid set of constraints
            while len(pairs) < k:
                index = Random_index(self.n)
                # pick a random position from the neighbors
                neighbor = random.choice(Neighbors(index, self.n))
                
                # if pair is seen already, then skip
                if {index, neighbor} in pairs: continue

                pairs.append({index, neighbor})

                self.C.append([index, neighbor] if self.S[index] < self.S[neighbor] 
                                                else [neighbor, index])
            
            # there is an inequality cycle in the generated constraints. The puzzle may still 
            # have a solution, but we will like to avoid this situation.
            if nx.is_directed_acyclic_graph(nx.DiGraph(self.C)):
                break        

        # mask the solved puzzle, keeping only r (random) positions revealed
        revealed = []

        # simulating sampling with replacement, to ensure 
        # there are r unique positions 
        while len(set(revealed)) != r:
            revealed.append(Random_index(self.n))
        
        # the unsolved puzzle will have unrevealed positions masked as 0
        self.U = np.zeros((self.n, self.n), dtype = int)
        for index in revealed:
            self.U[index] = self.S[index]

        # ?? Check comptabitibility of revealed indices and inqualities -- both
        # sides of an inequality should not be revealed. Looks bad.  
    
    def Solve(self, algorithm = "logic_solver"):
        '''Solves the puzzle using the given algorithm.
        '''
        if algorithm == "vanilla_backtrack":
            self.S = Vanilla_backtrack(deepcopy(self.U), self.C)

        elif algorithm == "logic_solver":
            self.S = Logic_solver(deepcopy(self.U), self.C)

        elif algorithm == "logic_solver_deterministic":
            self.S = Logic_solver(deepcopy(self.U), self.C, randomize = False)
            
        else:
            raise ValueError("Algorithm not supported.")

    def Print(self, board = "both", inequalities = False):
        '''Prints the puzzle in a readable format.
        '''
        C = None # by default, no inequalities are printed
        if inequalities:
            C = self.C # print inequalities

        if board == "both" or board == "unsolved":
            # print unsolved puzzle
            if self.U is None: print("\nPuzzle not generated yet.")
            else: print("\nUnsolved puzzle: \n%s"%Display_grid(self.U, C))

        if board == "both" or board == "solved":
            # print solved puzzle
            if self.S is None: print("\nPuzzle not solved yet.")
            elif self.S is False: print("\nPuzzle is not solvable.")
            else: print("\nSolved puzzle: \n%s"%Display_grid(self.S, C))