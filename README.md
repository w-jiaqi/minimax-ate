# minimax-ate

**https://minimax-ate.onrender.com**

Minimax Estimation of the Average Treatment Effect (ATE)

- `minimax_finder.py` — solver: finds the minimax estimator via convex optimization
- `cube_minimax_vs_dim.py` — compares minimax vs difference-in-means risk on random states
- `make_states_heatmap.py` — plots a heatmap of the risk gap as a function of |ATE| and outcome heterogeneity

## running

```bash
pip install numpy cvxpy matplotlib
```

```bash
# solve minimax estimator for N=10, balanced design
python minimax_finder.py --N 10

# uniform design treating m=4 units
python minimax_finder.py --N 10 --m 4

# compare minimax vs DiM on 100 random states
python cube_minimax_vs_dim.py --minimax-file minimax_finder.py --N 10 --num-states 100

# generate heatmap
python make_states_heatmap.py --minimax-file minimax_finder.py --N 12 --plot-out heatmap.png
```
