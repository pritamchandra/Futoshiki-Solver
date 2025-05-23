
## Futoshiki Solver

The project generates instances of the [Futoshiki](https://www.futoshiki.com/) puzzle and builds programs to solve it. A puzzle is instantiated as an object of the class `Futoshiki_puzzle`.

```python
P = Futoshiki_puzzle(n, S = None, C = None, U = None)
# n integer
# S, U are n x n integer matrices
# C is a list of pairs of tuples [[(i, j), (a, b)], ...]
```

**Parameters.** 
- `n`: denotes that the size of the puzzle is `n x n`.

- `U`: represents the unsolved state of the puzzle as an `n x n `matrix. `U` has entries in the set `{0, 1, ..., n}` where `0`s refer to hidden positions and every other entry is a revealed position. 

- `S`: solved state of the puzzle, that is, `S` agrees with `U` on the revealed positions, `S` is a latin square, and `S` satisfies the inequality constraints posed by `C`.

- `C`: a list of inequality constraints. If `[(i, j), (a, b)]` is in `C`, then `S[i, j] < S[a, b]`.

A puzzle can simply be generated by specifying `n`, eg. `P = Futoshiki_puzzle(n = 5)` and the rest of the paramteres can be generated by `P.Generate()`.

### Class Methods
```python
P.Generate(k = 3, r = 1)
# k, r integers
```
Post function call `P.S`, `P.U`, `P.C` are generated.

NOTE: the puzzles generated are not equipped to have unique solutions yet.

**Parameters.**
- `k`: the number of inequality constraints, or `len(C)`. The experiments are run with `k` roughly 30% of `n^2` (the number of entries in the grid).

- `r`: the number of revealed entries in the unsolved puzzle. In the experiments `r` is roughly 14% of `n^2`.


```python
P.Solve(algorithm = 'logic_solver')
# algorithm can also be 'logic_solver_deterministic', 'vanilla_backtrack'
```
Post function call the `algorithm` generates `P.S` as a function of `P.U` and `P.C`.

- `vanilla_backtrack` runs a brute force  seaerch algorithm on `P.U` to find `P.S` once all constraints are satisfied.

- `logic_solver` implements the logical techniques of elimination and insertion equipped with a backtracking strategy to find a solution with the minimum number of guesses. The solver is *randomized* in the sense that a guess is randomly picked from the logically possible values at any running state. The algorithm is briefly described in a later section _Brief Algorithm Description_.

- `logic_solver_deterministic` is a deterministic version of the above algorithm where the guesses are picked in ascending order.

`logic_solver` is significantly faster than `vanilla_backtrack`. A comparison graph is provided in the section _Speed Comparison_.

```python
P.Print(board = "both", inequalities = False)
# board can also be "unsolved" or "solved"
# inequalities can also be True
```
Prints the solved and the unsolved grid (or both), with or without the inequalities, as specified in the function call.

For example `P.Print(board = "unsolved", inequalities = True)` generates the following output. 
```
Unsolved puzzle: 

5  *  * > *  *
V     A      A
*  *  *   2  *
              
*  *  * < *  *
   A          
*  *  *   3  *
A  V      A   
*  2  *   *  1
```
Similarly, `P.Print(board = "solved", inequalities = False)` will output the following.
```
Solved puzzle: 

5 3 4 1 2
1 4 5 2 3
3 1 2 4 5
2 5 1 3 4
4 2 3 5 1
```
`P.Print()` also handles corner cases when the puzzle is not solved yet, or has no solution.

<hr>

### Brief Algorithm Descriptions

**Puzzle Generation approach.** \
Initially generate `P.S` as a [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix) and shuffle its rows and columns. Randomly select `k` *consistent* index pairs in `P.C` as inequality constraints. Generate `P.U` as all zeros, and randomly select `r` *consistent* positions to agree with `P.S`. 

**Strategies behind logic_solver.** \
The game imposes two kinds of constraints on a puzzle `P`. First the `Line` constraints which ensure that the solve grid is a latin square, that is, in each row and each column there is exactly one occurence of the entry, and the `Inequality` constraints which are imposed by `P.C`. 

In the algorithm, first a residue grid `R` is generated which holds the list of values currently available to an index. The following is an example of the residue grid at an arbitrary running instance.

```
      5      3,4    2,3,4 >   1    2,3,4
      V                 A              A
  1,3,4  1,3,4,5    3,4,5     2    3,4,5
                                        
1,2,3,4    1,3,4  1,2,3,4 < 4,5  2,3,4,5
               A                        
  1,2,4      4,5  1,2,4,5     3    2,4,5
      A        V              A         
    3,4        2    3,4,5   4,5        1
```
Now each of two kinds of constraints can be used to employ two kinds of operations on `R`, they are `elimination` and `insertion`. `elimination` is when the current state of the grid renders some values impossible in some locations, and `insertion` is when the residue grid enforces a location to have a certain entry.

Based on these observations, the method `Logic_solver(U, C)` (refer to [logic_solver.py](logic_solver.py)) is supplied by the following helper functions:
- `Inequality_elimination` runs only once in the beginning. It goes through every inequality chain (including trivial ones) and eliminates impossible values. For example a chain like `* < * < *` in a `5 x 5` grid would imply that the first entry cannot be `4` or `5`.

- `Inequality_insertion` runs repeatedly in the algorithm. Goes through every inequality and checks if there is a unique pair in the residue grid `R` that satisfies it. If such a pair exists it enforces an insertion.

- `Line_elimination` runs initially on the revealed entries, and then runs repeatedly whenever a value is inserted or solved. For every solved entry it removes the entry from the residue in the row and column.

- `Line_insertion` runs throughout the program by going through unsolved indices in every row and column and checking if there is a unique residue, in which case an insertion is enforced. 

Once the puzzle is reduced with the above techniques, but still a complete solution cannot be enforced, we call the `Backtrack_on_residue` function. This is an updated backtracking method equipped with the above strategies so that the number of guesses can be minimized. The function (randomly or deterministically, as specified) inserts an guess from the possible values in the residue grid and immediately reduces the residue grid `R` using the above strategies, and calls itself recursively. If a solution is not reached it makes another guess. 

<hr>

### Speed comparison Logic_solver vs Vanilla_backtrack

The following bar graph provides a run time comparison of the `Logic_solver` algorithm with the `Vanilla_backtrack` algorithm. For each `n` in the set `{4, ..., 14}`, we generate `50` puzzles and solve each of them with the two algorithms. The puzzles are generated with `k` and `r` roughly 30% and 14% if `n^2`.

The bars denote how many of those puzzles ran within `5s` for each algorithm. On the head of each bar we print two more values. 

The `red` value indicates the average run time of the solver function for the `< 5s` runs. 

The `blue` value indicates the average number of guesses the algorithm had to make before reaching a solution, in multiples of `1000`.

 Note that for each `n`, the average is computed with different normalizers (height of the bar) for two solvers, as the number of `< 5s` runs are different. 

![test](performance_plot.png "Title")

The bar graph is generated by the file [performance_plot](performance_plot.py).
<hr>

### Future Work
1. Generate puzzles with unique solutions.
2. Optimize the inequality chain finding algorithm.
3. Create a GUI to visualize the elimination and insertion techniques.
4. Update the solver to be able to generate hints.
5. Create a GUI to play the puzzle with hints. 
