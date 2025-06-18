# ---------------------------------------------------------------------
#  Part 3 – Estimation from sparse (48-month) observations
# ---------------------------------------------------------------------

import numpy as np
from numpy.random import default_rng

rng = default_rng(seed=42)          

# ---------- TRUE CTMC SPECIFICATION (same as Part 2) -----------------
Q_true = np.array([
    [-0.0085, 0.0050, 0.0025, 0.0000, 0.0010],
    [ 0.0000,-0.0140, 0.0050, 0.0040, 0.0050],
    [ 0.0000, 0.0000,-0.0080, 0.0030, 0.0050],
    [ 0.0000, 0.0000, 0.0000,-0.0090, 0.0090],
    [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   
])
n_states   = Q_true.shape[0]
DEATH      = n_states - 1                  

# Helper: embedded jump-probability matrix for any given Q --------------
def embedded_P(Q):
    P = np.zeros_like(Q)
    for i in range(n_states):
        if i == DEATH:
            P[i, i] = 1.0
        else:
            P[i] = Q[i].copy()
            P[i, i] = 0.0
            P[i] /= -Q[i, i]
    return P

# ---------------------------------------------------------------------
# Task 12 – generate sparse observational data
# ---------------------------------------------------------------------
def simulate_patient(Q, screen_interval=48.0):
    """Return list of observed states at t = 0, 48, 96, … until first 5."""
    P = embedded_P(Q)
    t           = 0.0
    next_screen = 0.0
    state       = 0           
    obs         = [state]        

    while state != DEATH:
        # simulate one sojourn
        rate   = -Q[state, state]
        wait   = rng.exponential(1/rate)
        t     += wait

        # jump
        state  = rng.choice(n_states, p=P[state])

        # record any missed screens that occur before the jump
        while next_screen + screen_interval <= t:
            next_screen += screen_interval
            # if patient died before this screening, we only discover death now
            obs.append(DEATH if state == DEATH else state)

        # if jump lands exactly on a screening instant
        if abs(t - (next_screen + screen_interval)) < 1e-12:
            next_screen += screen_interval
            obs.append(state)

    # ensure final recorded value is 5 (death) – might already be appended
    if obs[-1] != DEATH:
        obs.append(DEATH)
    return obs

N_patients  = 1000
screen_gap  = 48.0
observed_TS = [simulate_patient(Q_true, screen_gap) for _ in range(N_patients)]

# Example: print the first 3 observation vectors
print("Task 12 – first three observed time-series (indices 0-4 ≡ states 1-5):")
for i, ts in enumerate(observed_TS[:3], 1):
    print(f"  Patient {i}: {ts}")

# ---------------------------------------------------------------------
# Task 13 – Monte-Carlo EM to estimate Q from (Y₁,…,Yₙ)
# ---------------------------------------------------------------------
def MCEM(observations, dt=48.0, max_iter=50, tol=1e-3):
    """
    observations  : list of lists  (each list is the discrete 48-mth scan path)
    dt            : screening interval
    Returns Q_hat and convergence history of ||ΔQ||∞.
    """
    # ----------  initial guess Q⁽⁰⁾  (weakly informative) --------------
    Q = np.array([
        [-0.01,  0.003, 0.003, 0.000, 0.004],
        [ 0.000,-0.02 , 0.004, 0.005, 0.011],
        [ 0.000, 0.000,-0.01 , 0.004, 0.006],
        [ 0.000, 0.000, 0.000,-0.012, 0.012],
        [ 0.000, 0.000, 0.000, 0.000, 0.000]
    ])

    history = []
    for it in range(max_iter):
        P     = embedded_P(Q)
        Nij   = np.zeros_like(Q)        
        Si    = np.zeros(n_states)      

        # ---------- E-step: simulate complete paths -------------------
        for path in observations:
            for k in range(len(path) - 1):
                s0, s1 = path[k], path[k+1]

                # rejection-sampling bridge over one 48-month gap -------
                while True:
                    t      = 0.0
                    state  = s0
                    last_t = 0.0
                    local_N = np.zeros_like(Q)
                    local_S = np.zeros(n_states)

                    while t < dt:
                        if state == DEATH:
                            local_S[DEATH] += (dt - t)
                            t = dt
                            break

                        rate  = -Q[state, state]
                        wait  = rng.exponential(1/rate)
                        if t + wait >= dt:   
                            local_S[state] += (dt - t)
                            t = dt
                            break

                        # jump occurs
                        local_S[state] += (wait)
                        t += wait
                        next_state      = rng.choice(n_states, p=P[state])
                        local_N[state, next_state] += 1
                        state           = next_state

                    # accept if terminal state matches required observation
                    if state == s1:
                        Nij += local_N
                        Si  += local_S
                        break           

        # ---------- M-step: update Q -----------------------------------
        Q_new = Q.copy()
        for i in range(n_states):
            if i == DEATH:
                Q_new[i] = 0.0
                continue
            if Si[i] > 0:
                Q_new[i, :] = Nij[i] / Si[i]
                Q_new[i, i] = -np.sum(Q_new[i, :])
            else:  
                pass

        delta = np.max(np.abs(Q_new - Q))
        history.append(delta)
        Q = Q_new
        print(f"Iteration {it+1:2d}:  ||ΔQ||_∞ = {delta:.4e}")
        if delta < tol:
            break
    return Q, history

Q_hat, deltas = MCEM(observed_TS, dt=screen_gap)
print("\nTask 13 – Estimated transition-rate matrix Q̂:")
np.set_printoptions(precision=4, suppress=True)
print(Q_hat)

print("\nInfinity-norm distances per iteration:", deltas)
