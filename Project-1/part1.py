import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

np.random.seed(42)

# Transition matrix P for discrete-time model (states 1-5 mapped to indices 0-4)
P = np.array([
    [0.9915, 0.0050, 0.0025, 0.0,    0.001],
    [0.0,    0.9860, 0.0050, 0.0040, 0.005],
    [0.0,    0.0,    0.9920, 0.0030, 0.005],
    [0.0,    0.0,    0.0,    0.9910, 0.009],
    [0.0,    0.0,    0.0,    0.0,    1.0]
])

N = 1000  # number of women

# Containers for results
lifetimes = []
local_recurrences = []
state_at_120 = []

# --- Task 1: Simulate lifetimes, local recurrence, and record state at t=120 ---
for _ in range(N):
    current_state = 0  # start in state 1
    recurred_locally = False
    state120 = None

    for t in range(1, 10000):  # large max to ensure death
        next_state = np.random.choice(5, p=P[current_state])
        if next_state == 1:
            recurred_locally = True
        if t == 120:
            state120 = next_state
        current_state = next_state
        if current_state == 4:  # death reached
            lifetimes.append(t)
            local_recurrences.append(recurred_locally)
            state_at_120.append(state120 if state120 is not None else 4)
            break

# Convert results to arrays
lifetimes = np.array(lifetimes)
state_at_120 = np.array(state_at_120)
prop_local = np.mean(local_recurrences)

# Plot lifetime histogram (Task 1)
plt.figure(figsize=(8, 5))
plt.hist(lifetimes, bins=30, edgecolor='black')
plt.title('Lifetime Distribution After Surgery (1000 Simulations)')
plt.xlabel('Lifetime (months)')
plt.ylabel('Count')
plt.show()

print("Task 1: Proportion of women with local recurrence: {:.3f}".format(prop_local))


# --- Task 2: Distribution over states at t = 120 and χ² test ---
obs_counts = np.bincount(state_at_120, minlength=5)
obs_props = obs_counts / N

# Theoretical distribution at t=120
p0 = np.array([1, 0, 0, 0, 0])
P120 = np.linalg.matrix_power(P, 120)
theo_props = p0.dot(P120)

chi2_stat, p_val = chisquare(f_obs=obs_counts, f_exp=N * theo_props)

print("\nTask 2: State distribution at t=120 (simulated vs theoretical):")
for i in range(5):
    print(f"  State {i+1}: simulated {obs_props[i]:.3f}, theoretical {theo_props[i]:.3f}")
print(f"Chi-square test: statistic={chi2_stat:.2f}, p-value={p_val:.3f}")


# --- Task 3: Lifetime distribution vs. discrete phase-type ---
Ps = P[:4, :4]
ps = P[:4, 4]
pi = np.array([1, 0, 0, 0])

max_t = lifetimes.max()
theo_pmf = np.array([pi.dot(np.linalg.matrix_power(Ps, t-1)).dot(ps)
                     for t in range(1, max_t+1)])
obs_counts_life = np.bincount(lifetimes, minlength=max_t+1)[1:]

# Account for tail beyond max_t
tail_prob = 1 - theo_pmf.sum()
f_obs_adj = np.append(obs_counts_life, 0)
f_exp_adj = np.append(N * theo_pmf, N * tail_prob)

chi2_life, p_life = chisquare(f_obs=f_obs_adj, f_exp=f_exp_adj)

print("\nTask 3: Lifetime distribution goodness-of-fit to discrete phase-type:")
print(f"Tail probability beyond max observed lifetime: {tail_prob:.4f}")
print(f"Chi-square test: statistic={chi2_life:.2f}, p-value={p_life:.3f}")


# --- Task 4: E[lifetime | survive ≥12 & recurrence by 12] ---
def simulate_patient():
    """Simulate until absorption; return (lifetime, history list of states)."""
    history = []
    state = 0
    t = 0
    while state != 4:
        t += 1
        state = np.random.choice(5, p=P[state])
        history.append(state)
    return t, history

N_target = 1000
lifetimes_task4 = []
while len(lifetimes_task4) < N_target:
    T, hist = simulate_patient()
    if T >= 12 and any(s in (1, 2) for s in hist[:12]):
        lifetimes_task4.append(T)

print("\nTask 4: E[lifetime | survive ≥12 & recurrence by 12] =",
      np.mean(lifetimes_task4))


# --- Task 5: Fraction dying ≤350 months with control variate ---
n = 200      # patients per replicate
reps = 100   # number of replicates
frac_crude = []
mean_lifetimes = []

# theoretical mean lifetime E[T]
I = np.eye(4)
theoretical_mean = pi @ np.linalg.inv(I - Ps) @ np.ones(4)

for _ in range(reps):
    times = [simulate_patient()[0] for _ in range(n)]
    frac_crude.append(np.mean([t <= 350 for t in times]))
    mean_lifetimes.append(np.mean(times))

frac_crude = np.array(frac_crude)
mean_lifetimes = np.array(mean_lifetimes)

# optimal control variate coefficient
b_opt = np.cov(frac_crude, mean_lifetimes, ddof=1)[0,1] / np.var(mean_lifetimes, ddof=1)

# control variate estimator
est_cv = frac_crude + b_opt * (theoretical_mean - mean_lifetimes)

var_crude = np.var(frac_crude, ddof=1)
var_cv    = np.var(est_cv, ddof=1)
reduction = 100 * (var_crude - var_cv) / var_crude

print("\nTask 5: Var(crude) = {:.6f}, Var(CV) = {:.6f}, reduction = {:.1f}%".format(
      var_crude, var_cv, reduction))
