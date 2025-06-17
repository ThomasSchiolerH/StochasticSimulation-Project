import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest
from scipy.linalg import expm

np.random.seed(42)

# --- Task 7: Continuous-time simulation, summary statistics, distant recurrence ---
# Transition-rate matrix Q (continuous-time)
Q_cont = np.array([
    [-0.0085, 0.0050, 0.0025, 0.0,    0.001],
    [ 0.0,   -0.014,  0.005,  0.004,  0.005],
    [ 0.0,    0.0,   -0.008,  0.003,  0.005],
    [ 0.0,    0.0,    0.0,   -0.009,  0.009],
    [ 0.0,    0.0,    0.0,    0.0,    0.0  ]
])

# Build embedded jump probabilities
P_cont = np.zeros_like(Q_cont)
for i in range(Q_cont.shape[0] - 1):
    rates = Q_cont[i].copy()
    rates[i] = 0
    P_cont[i] = rates / (-Q_cont[i, i])
P_cont[-1, -1] = 1.0  # absorbing

N = 1000
lifetimes = []
distant_by_30_5 = []

for _ in range(N):
    t, state = 0.0, 0
    recurred_dist = False
    while state != 4:
        rate = -Q_cont[state, state]
        t += np.random.exponential(scale=1/rate)
        next_state = np.random.choice(5, p=P_cont[state])
        if next_state == 2 and t <= 30.5:
            recurred_dist = True
        state = next_state
    lifetimes.append(t)
    distant_by_30_5.append(recurred_dist)

lifetimes = np.array(lifetimes)
prop_distant = np.mean(distant_by_30_5)

# Mean & 95% CI for mean lifetime
mean_life = lifetimes.mean()
std_life = lifetimes.std(ddof=1)
se_life = std_life / np.sqrt(N)
z = 1.96
ci_mean = (mean_life - z*se_life, mean_life + z*se_life)

# 95% CI for standard deviation
df = N - 1
alpha = 0.05
ci_std = (
    std_life * np.sqrt(df / chi2.ppf(1 - alpha/2, df)),
    std_life * np.sqrt(df / chi2.ppf(alpha/2, df))
)

print("Task 7:")
print(f"Mean lifetime: {mean_life:.3f} [{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]")
print(f"Std dev: {std_life:.3f} [{ci_std[0]:.3f}, {ci_std[1]:.3f}]")
print(f"Proportion with distant recurrence by 30.5 months: {prop_distant:.3f}")

plt.figure()
plt.hist(lifetimes, bins=30, edgecolor='k')
plt.title('Task 7: Lifetime Distribution (CTMC)')
plt.xlabel('Lifetime (months)')
plt.ylabel('Count')
plt.show()

# --- Task 8: Goodness-of-fit to continuous phase-type distribution (KS test) ---
Qs = Q_cont[:4, :4]
ones = np.ones(4)
p0 = np.array([1, 0, 0, 0])

def cdf_continuous(t):
    t = np.atleast_1d(t)
    vals = []
    for ti in t:
        vals.append(1 - p0.dot(expm(Qs * ti)).dot(ones))
    return np.array(vals)

ks_stat, ks_p = kstest(lifetimes, cdf_continuous)
print("\nTask 8:")
print(f"KS statistic: {ks_stat:.3f}, p-value: {ks_p:.3f}")

# --- Task 9: Kaplan–Meier curves for treated vs. untreated ---
# New Q for treated
Q_treated = np.array([
    [-(0.0025+0.00125+0.001), 0.0025,  0.00125, 0.0,   0.001],
    [0.0,                   -(0.002 + 0.005), 0.0,    0.002, 0.005],
    [0.0,                     0.0,   -(0.003 + 0.005), 0.003, 0.005],
    [0.0,                     0.0,    0.0,   -0.009, 0.009],
    [0.0,                     0.0,    0.0,    0.0,   0.0  ]
])
P_treated = np.zeros_like(Q_treated)
for i in range(4):
    rates = Q_treated[i].copy()
    rates[i] = 0
    P_treated[i] = rates / (-Q_treated[i, i])
P_treated[-1, -1] = 1.0

# Simulate treated cohort
lifetimes_treated = []
for _ in range(N):
    t, state = 0.0, 0
    while state != 4:
        rate = -Q_treated[state, state]
        t += np.random.exponential(scale=1/rate)
        state = np.random.choice(5, p=P_treated[state])
    lifetimes_treated.append(t)
lifetimes_treated = np.array(lifetimes_treated)

# Kaplan–Meier estimator
def kaplan_meier(times):
    times = np.sort(times)
    n = len(times)
    uniq, counts = np.unique(times, return_counts=True)
    at_risk = n
    surv = 1.0
    surv_times, surv_probs = [], []
    for t, d in zip(uniq, counts):
        surv *= (1 - d/at_risk)
        surv_times.append(t)
        surv_probs.append(surv)
        at_risk -= d
    return np.array(surv_times), np.array(surv_probs)

t_unt, s_unt = kaplan_meier(lifetimes)
t_tr,  s_tr  = kaplan_meier(lifetimes_treated)

plt.figure()
plt.step(t_unt, s_unt, where='post', label='Untreated')
plt.step(t_tr,  s_tr,  where='post', label='Treated')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.title('Task 9: Kaplan–Meier Survival Curves')
plt.legend()
plt.show()

# --- Task 10: Log-rank test for treated vs. untreated survival ----------------

from scipy.stats import chi2

def logrank_test(times1, times2):
    """
    Perform a two‐sample log‐rank test (no censoring assumed).
    times1, times2: arrays of event times for group1 and group2.
    Returns (chi2_stat, p_value).
    """
    # Combine all distinct event times
    times_all = np.sort(np.unique(np.concatenate([times1, times2])))
    n1 = len(times1)
    n2 = len(times2)
    O1 = 0.0  # observed events in group1
    E1 = 0.0  # expected events in group1
    V1 = 0.0  # variance of O1

    # At each event time t:
    for t in times_all:
        # number at risk just before t
        r1 = np.sum(times1 >= t)
        r2 = np.sum(times2 >= t)
        r = r1 + r2
        # number of events at t
        d1 = np.sum(times1 == t)
        d2 = np.sum(times2 == t)
        d = d1 + d2
        if r > 0:
            # expected events in group1 at time t
            e1 = d * (r1 / r)
            # variance term
            v1 = (r1 * r2 * d * (r - d)) / (r**2 * (r - 1)) if r > 1 else 0.0

            O1 += d1
            E1 += e1
            V1 += v1

    # compute chi‐square statistic
    chi2_stat = (O1 - E1)**2 / V1 if V1 > 0 else np.nan
    p_value = 1 - chi2.cdf(chi2_stat, df=1) if V1 > 0 else np.nan
    return chi2_stat, p_value

chi2_stat, p_val = logrank_test(lifetimes, lifetimes_treated)
print("\nTask 10:")
print(f"Log‐rank test: χ² = {chi2_stat:.3f}, p = {p_val:.3f}")

# --- Task 11: Extending to Erlang sojourn times -------------------------------

# Here we replace each exponential sojourn time by Erlang(k, rate).
# Example: shape k=2 for all states.
k = 2
lifetimes_erlang = []

for _ in range(N):
    t, state = 0.0, 0
    while state != 4:
        rate = -Q_cont[state, state]
        # Erlang(k, rate): sum of k independent Exp(rate)
        sojourn = np.sum(np.random.exponential(scale=1/rate, size=k))
        t += sojourn
        state = np.random.choice(5, p=P_cont[state])
    lifetimes_erlang.append(t)

lifetimes_erlang = np.array(lifetimes_erlang)
print("\nTask 11:")
print(f"Mean lifetime with Erlang(k={k}) sojourn: {lifetimes_erlang.mean():.3f} months")

plt.figure()
plt.hist(lifetimes_erlang, bins=30, edgecolor='k')
plt.title(f'Task 11: Lifetime Distribution with Erlang(k={k}) Sojourns')
plt.xlabel('Lifetime (months)')
plt.ylabel('Count')
plt.show()
