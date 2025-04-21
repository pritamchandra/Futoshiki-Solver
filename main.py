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
from puzzle_class import *
from logic_solver import *

np.set_printoptions(legacy = '1.25')

# # _____________________ DO NOT change the following block

Vanilla_backtrack.calls = 0
Backtrack_on_residue.calls = 0

# # ________________________________________________________

# Usage example

n = 11 # size of the puzzle
P = Futoshiki_puzzle(n = n)
# If the user doesn't want the puzzle to be generated, they can also 
# provide S, U, C in Futoshiki_puzzle(n, S = S, U = U, C = C) which represent
# the solved board, the unsolved baord and the constraints respectively.

# Generate a puzzle with k constraints and r pre-assigned cells
# This will generate P.S, P.U and P.C
P.Generate_puzzle(k = (n ** 2)//3, 
                  r = (n ** 2)//7)

# Print the solved board 
P.Print(board = "unsolved", inequalities = True)
# Print() can be called with board = "solved" or "unsolved" or "both", 
# and inequalities = True or False. 

P.Solve(algorithm = 'logic_solver')
# P.Solve() can be called wit algorithm = 'logic_solver_deterministic' or 'vanilla_backtrack'.

P.Print(board = "solved", inequalities = True)

# print(Is_valid(P.S, P.C)) # post solving check if the solution is valid 