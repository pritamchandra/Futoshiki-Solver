# Warning! Long running code

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

# # _______________________________________________________________

# helper method
def Extract_plottable_values(Data):
    '''Given a list of tuples (n, [(time, calls), ...]), 
    this function extracts the time and call values for plotting.
    '''
    X = []
    rate = []
    time = []
    calls = []
    epoch = len(Data[0][1])

    for data in Data:
        X.append(data[0])
        rate.append(np.sum(data[1] == 0))
        if epoch - rate[-1] == 0:
            time.append(np.inf)
            calls.append(np.inf)
            continue

        time.append(np.sum(data[1]) / (epoch - rate[-1]))
        calls.append(np.sum(data[2]) / (epoch - rate[-1]))
        
    return [X, rate, time, calls]

# # _______________________________________________________________

# Build the data-set
timeout = 5 # seconds
epochs = 50

# # Data = [vanilla_info, logic_info] # either use this data generated earlier or run the following code
# # NOTE: the code can take up to 1 hour to run
'''
Data = [[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
         [0, 0, 0, 0, 0, 9, 17, 34, 42, 50, 49, 50, 50], 
         [0.0002466011047363281, 0.0006818532943725586, 0.0026663780212402345, 0.024431638717651367, 0.13658662319183348, 0.5506198871426466, 8.2687756653988, 1.5228297263383865, 2.617774486541748, np.inf, 2.765511989593506, np.inf, np.inf], 
         [7.92, 32.86, 148.48, 1539.9, 7884.54, 27064.024390243903, 53580.72727272727, 79258.9375, 166027.75, np.inf, 92104.0, np.inf, np.inf]],
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
         [0, 0, 0, 0, 0, 0, 6, 17, 23, 25, 41, 41, 44], 
         [0.00023950099945068359, 0.0005583477020263672, 0.0013272809982299806, 0.0033721303939819335, 0.0067474365234375, 0.053237590789794925, 0.1872912807898088, 0.2628524231188225, 0.2250267576288294, 0.47694902420043944, 0.3578384717305501, 0.5894114971160889, 2.400238275527954], 
         [0.0, 0.34, 2.18, 6.48, 12.24, 128.8, 386.02272727272725, 427.06060606060606, 282.037037037037, 472.24, 368.1111111111111, 395.55555555555554, 1490.8333333333333]]]

vanilla_info, logic_info = Data         
'''

vanilla_info = []
logic_info = []
False_count = 0


for n in range(2, 15):
    vanilla_times = []
    vanilla_calls = []

    logic_times = []
    logic_calls = []

    # Loop through epochs to gather performance data
    for _ in range(epochs):
        P = Futoshiki_puzzle(n = n)
        P.Generate_puzzle(k = (n ** 2)//3, r = (n ** 2)//7)

        Vanilla_backtrack.calls = 0
        P.S = None
        try:
            t = time()
            func_timeout(timeout, P.Solve, args = ("vanilla_backtrack",))
            vanilla_times.append(time() - t)
            vanilla_calls.append(Vanilla_backtrack.calls)
         
        except:
            vanilla_times.append(0)
            vanilla_calls.append(0)
        

        Backtrack_on_residue.calls = 0
        P.S = None
        try:
            t = time()
            func_timeout(timeout, P.Solve, args = ("logic_solver",))
            logic_times.append(time() - t)
            logic_calls.append(Backtrack_on_residue.calls)
    
            if P.S is False or Is_valid(P.S, P.C) is False:
                False_count += 1
        except:
            logic_times.append(0)
            logic_calls.append(0)
        
    vanilla_info.append((n, np.array(vanilla_times), np.array(vanilla_calls)))
    logic_info.append((n, np.array(logic_times), np.array(logic_calls)))

full_vanilla_data = Extract_plottable_values(vanilla_info)
full_logic_data = Extract_plottable_values(logic_info)

# remove the first two entries from every list in the data, to remove the trivial cases
vanilla_data = [None] * len(full_vanilla_data)
logic_data = [None] * len(full_logic_data)

for i in range(len(full_vanilla_data)):
    vanilla_data[i] = full_vanilla_data[i][2:]
    logic_data[i] = full_logic_data[i][2:]

# # _______________________________________________________________
# Plot the data


X = vanilla_data[0]
N = len(vanilla_data[0])

alt_X = np.array([X, X]).T.flatten()
alt_failures = np.array([logic_data[1], vanilla_data[1]]).T.flatten()
alt_successes = 50 - alt_failures
alt_times = np.array([logic_data[2], vanilla_data[2]]).T.flatten().round(2)
alt_calls = (np.array([logic_data[3], vanilla_data[3]]).T.flatten()/1000).round(2)

plt.figure(figsize = (10, 4.8))

df = pd.DataFrame({
    'X' : alt_X,
    '# successes' : alt_successes,
    'Algo.': ['Logic_solver', 'Vanilla_backtrack'] * N,
})


sns.barplot(data = df, 
            x = 'X', 
            y = '# successes', 
            hue = 'Algo.', 
            palette = ['#4F7942', '#4F4F4F'],
            width = .8,)


plt.title('Comparative performance of the algorithms.', fontweight = 'bold', 
                                                        family = 'monospace', 
                                                        fontsize = 11)

plt.xlabel('Grid size' + r'$\longrightarrow$', fontweight = 'bold', 
                                               family = 'monospace')

plt.ylabel('Number of <5s runs in 50 epochs' + r'$\longrightarrow$', fontweight = 'bold', 
                                                                     family = 'monospace')

plt.tight_layout()
plt.ylim(0, 70) # increase y axis limit

# plot legends
plt.legend(loc = 'upper right')

plt.text(9.65, 52, 'avg runtime(s)', rotation = 'horizontal',
                                        ha = 'center', 
                                        va = 'bottom', 
                                        fontsize = 11,
                                        family = 'monospace',
                                        color = '#800000',
                                        fontstyle = 'normal',
                                        fontweight = 'bold',)

plt.text(9.45, 55, 'avg #guesses/10^3', rotation = 'horizontal',
                                        ha = 'center', 
                                        va = 'bottom', 
                                        fontsize = 11,
                                        family = 'monospace',
                                        color = '#00008B',
                                        fontstyle = 'normal',
                                        fontweight = 'bold',)

# plot run times and the number of calls in the bar graph
x_0, x_1  = -0.20, 0.23; parity = 0 # this values are particular to the plot and the configuration
x = x_0 
for index, value in enumerate(alt_successes):
    # plot run times of the two algorithms 
    plt.text(x, value + 0.5, str(alt_times[index]), rotation = 'horizontal', 
                                                        ha = 'center', 
                                                        va = 'bottom', 
                                                        fontsize = 9,
                                                        family = 'monospace',
                                                        color = '#800000',
                                                        fontstyle = 'normal',
                                                        fontweight = 'bold',)
    
    # plot number of calls of the two algorithms
    plt.text(x, value + 3.5, str(alt_calls[index]), rotation = 'horizontal',
                                                        ha = 'center', 
                                                        va = 'bottom', 
                                                        fontsize = 8,
                                                        family = 'monospace',
                                                        color = '#00008B',
                                                        fontstyle = 'normal',
                                                        fontweight = 'bold',)
    
    if parity == 0:
        x_0 += 1; parity = 1; x = x_1

    else:
        x_1 += 1; parity = 0; x = x_0

plt.savefig("performance_plot.png", dpi = 300, bbox_inches = 'tight')
plt.show()
