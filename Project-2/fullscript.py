import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from itertools import count

# -------------------------
# Simulation Engine Module
# -------------------------
def simulate_patient_flow(bed_config, arrival_rates, service_rates, relocation_probs,
                          urgency_points=None, sim_time=365, seed=None,
                          service_dist='exponential', lognormal_var=None,
                          record_interval=1.0):
    """
    Simulate patient flow in hospital.

    bed_config: dict ward->int capacity
    arrival_rates: dict ward->λ (per day)
    service_rates: dict ward->μ (per day)
    relocation_probs: dict ward_i->dict ward_j->p_ij
    sim_time: simulation horizon (days)
    service_dist: 'exponential' or 'lognormal'
    lognormal_var: dict ward->variance (for lognormal)
    record_interval: time between occupancy records
    """
    rng = np.random.default_rng(seed)
    event_queue = []
    ctr = count()

    wards = list(bed_config.keys())
    occupancy  = {w: 0 for w in wards}
    admissions = {w: 0 for w in wards}
    relocations = {w: 0 for w in wards}
    losses      = {w: 0 for w in wards}

    # Schedule initial arrivals
    for w in wards:
        lam = arrival_rates.get(w, 0)
        if lam > 0:
            t0 = rng.exponential(1/lam)
            heapq.heappush(event_queue, (t0, next(ctr), 'arrival', w))

    # Prepare recording
    record_times = np.arange(0, sim_time + record_interval, record_interval)
    occ_records   = {w: [] for w in wards}
    time_records  = []
    next_rec_idx  = 0
    current_time  = 0.0

    while event_queue:
        t, _, ev, ward = heapq.heappop(event_queue)
        if t > sim_time:
            break
        current_time = t

        # Record occupancy up to now
        while next_rec_idx < len(record_times) and record_times[next_rec_idx] <= current_time:
            time_records.append(record_times[next_rec_idx])
            for w in wards:
                occ_records[w].append(occupancy[w])
            next_rec_idx += 1

        # Skip if ward not in this config
        if ward not in wards:
            continue

        if ev == 'arrival':
            lam = arrival_rates.get(ward, 0)
            if lam > 0:
                t_next = current_time + rng.exponential(1/lam)
                heapq.heappush(event_queue, (t_next, next(ctr), 'arrival', ward))

            # Admit if bed available
            if occupancy[ward] < bed_config[ward]:
                admissions[ward] += 1
                occupancy[ward] += 1
                los = _draw_los(rng, service_rates[ward], service_dist, lognormal_var, ward)
                heapq.heappush(event_queue, (current_time + los, next(ctr), 'departure', ward))
            else:
                # Relocate
                pmat = relocation_probs.get(ward, {})
                targets = [j for j,p in pmat.items() if j in wards and p > 0]
                probs   = [pmat[j] for j in targets]
                if targets and sum(probs) > 0:
                    probs = np.array(probs) / np.sum(probs)
                    j = rng.choice(targets, p=probs)
                    relocations[ward] += 1
                    if occupancy[j] < bed_config[j]:
                        admissions[j] += 1
                        occupancy[j] += 1
                        los = _draw_los(rng, service_rates[ward], service_dist, lognormal_var, ward)
                        heapq.heappush(event_queue, (current_time + los, next(ctr), 'departure', j))
                    else:
                        losses[ward] += 1
                else:
                    losses[ward] += 1

        elif ev == 'departure':
            occupancy[ward] = max(0, occupancy[ward] - 1)

    # Final occupancy record
    while next_rec_idx < len(record_times):
        time_records.append(record_times[next_rec_idx])
        for w in wards:
            occ_records[w].append(occupancy[w])
        next_rec_idx += 1

    return {
        'time': np.array(time_records),
        'occupancy': pd.DataFrame(occ_records, index=time_records),
        'admissions': admissions,
        'relocations': relocations,
        'losses': losses
    }

def _draw_los(rng, mu, dist, lognorm_var, ward):
    if dist == 'exponential':
        return rng.exponential(1/mu)
    var = lognorm_var[ward]
    mean = 1/mu
    # compute lognormal parameters
    mu_logn = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma_logn = np.sqrt(np.log(1 + var/mean**2))
    return rng.lognormal(mu_logn, sigma_logn)

# -------------------------
# F* Bed Allocation Module
# -------------------------
def allocate_beds_for_F(total_beds, base_config, arrival_rates, service_rates,
                        relocation_probs, urgency_points, target_F_rate=0.95,
                        sim_time=365, seed=42):
    """
    Greedily reallocate beds to ward 'F' until type-F admission >= target_F_rate.
    """
    config = base_config.copy()
    config['F'] = 1
    removed_from = {w: 0 for w in base_config}

    while True:
        rates = arrival_rates.copy(); rates['F'] = arrival_rates['F']
        mus   = service_rates.copy(); mus['F'] = service_rates['F']
        probs = relocation_probs.copy()
        res = simulate_patient_flow(config, rates, mus, probs,
                                    sim_time=sim_time, seed=seed)
        F_adm = res['admissions']['F']
        F_arr = sim_time * arrival_rates['F']
        rate  = F_adm / F_arr
        if rate >= target_F_rate:
            break

        # remove one bed from lowest urgency-per-bed ward
        ratios = {w: urgency_points[w]/config[w] for w in base_config if config[w] > 1}
        w_rem  = min(ratios, key=ratios.get)
        config[w_rem] -= 1
        config['F'] += 1
        removed_from[w_rem] += 1

    return config, removed_from, rate

# -------------------------
# Metrics Module
# -------------------------
def compute_metrics(sim_result, bed_config, arrival_rates):
    admissions  = sim_result['admissions']
    relocations = sim_result['relocations']
    losses      = sim_result['losses']
    occ         = sim_result['occupancy']

    data = []
    for w in bed_config:
        total_arr = admissions[w] + relocations[w] + losses[w]
        adm_prob   = admissions[w] / total_arr if total_arr>0 else 0
        exp_reloc  = relocations[w]
        full_prob  = np.mean(occ[w] >= bed_config[w])
        data.append((w, adm_prob, exp_reloc, full_prob))

    df = pd.DataFrame(data, columns=['ward','adm_prob','exp_relocations','full_prob'])
    return df.set_index('ward')

# -------------------------
# Sensitivity Analysis
# -------------------------
def sensitivity_analysis(bed_configs, arrival_rates, service_rates,
                         relocation_probs, urgency_points,
                         sim_time=365, seed=42):
    results = {}
    # Exponential LOS
    for name, bc in bed_configs.items():
        res = simulate_patient_flow(bc, arrival_rates, service_rates,
                                    relocation_probs, sim_time=sim_time, seed=seed)
        results[(name, 'exp')] = compute_metrics(res, bc, arrival_rates)
    # Log-normal LOS with variances 2/μ²,3/μ²,4/μ²
    for vf in [2,3,4]:
        var = {w: vf / service_rates[w]**2 for w in service_rates}
        for name, bc in bed_configs.items():
            res = simulate_patient_flow(bc, arrival_rates, service_rates,
                                        relocation_probs,
                                        sim_time=sim_time, seed=seed,
                                        service_dist='lognormal',
                                        lognormal_var=var)
            results[(name, f'lognorm_{vf}')] = compute_metrics(res, bc, arrival_rates)
    return results

# -------------------------
# Plotting Module
# -------------------------
def plot_adm_vs_reloc(metrics_df):
    plt.figure()
    plt.scatter(metrics_df['adm_prob'], metrics_df['exp_relocations'])
    for w in metrics_df.index:
        plt.text(metrics_df.at[w,'adm_prob'], metrics_df.at[w,'exp_relocations'], w)
    plt.xlabel('Admission Probability')
    plt.ylabel('Expected Relocations')
    plt.title('Admissions vs. Relocations')
    plt.show()

def plot_full_prob(metrics_df):
    plt.figure()
    metrics_df['full_prob'].plot(kind='bar')
    plt.ylabel('Full Ward Probability')
    plt.title('Full Ward Probabilities')
    plt.show()

def plot_penalty_heatmap(metrics_dct, urgency_points):
    # Collect all scenarios & all wards present
    scenarios = sorted({scen for (_,scen) in metrics_dct.keys()})
    wards     = sorted({w for df in metrics_dct.values() for w in df.index})

    heat = pd.DataFrame(index=wards, columns=scenarios, dtype=float).fillna(0.0)
    for (_, scen), df in metrics_dct.items():
        for w in wards:
            if w in df.index:
                heat.at[w, scen] = urgency_points.get(w,0) * df.at[w, 'exp_relocations']
    plt.figure()
    plt.imshow(heat.values, aspect='auto')
    plt.colorbar(label='Urgency-weighted Penalty')
    plt.xticks(range(len(scenarios)), scenarios, rotation=45)
    plt.yticks(range(len(wards)), wards)
    plt.title('Urgency-weighted Penalties Heatmap')
    plt.tight_layout()
    plt.show()

# -------------------------
# LaTeX Summary Module
# -------------------------
def generate_latex_summary(metrics_dct, best_config, best_score):
    summary_tables = {}
    for (cfg, scen), df in metrics_dct.items():
        summary_tables[f"{cfg}_{scen}"] = df
    return {
        'tables': summary_tables,
        'best_config': best_config,
        'best_score': best_score
    }

# -------------------------
# Main Execution
# -------------------------
def main():
    # Base parameters
    bed_config = {'A':55, 'B':40, 'C':30, 'D':20, 'E':20}
    arrival_rates = {'A':14.5, 'B':11.0, 'C':8.0, 'D':6.5, 'E':5.0}
    service_rates = {'A':1/2.9, 'B':1/4.0, 'C':1/4.5, 'D':1/1.4, 'E':1/3.9}
    urgency_points = {'A':7, 'B':5, 'C':2, 'D':10, 'E':5}

    relocation_probs = {
        'A':{'B':0.05,'C':0.10,'D':0.05,'E':0.80},
        'B':{'A':0.20,'C':0.50,'D':0.15,'E':0.15},
        'C':{'A':0.30,'B':0.20,'D':0.20,'E':0.30},
        'D':{'A':0.35,'B':0.30,'C':0.05,'E':0.30},
        'E':{'A':0.20,'B':0.10,'C':0.60,'D':0.10},
        # F will be added below
    }

    # 1) Base simulation
    base_res = simulate_patient_flow(bed_config, arrival_rates, service_rates,
                                     relocation_probs, sim_time=365, seed=0)
    base_metrics = compute_metrics(base_res, bed_config, arrival_rates)
    print("Base case metrics:\n", base_metrics)

    # 2) Introduce F*
    arrival_rates['F'] = 13.0
    service_rates['F'] = 1/2.2
    # extend relocation matrix
    relocation_probs['F'] = {'A':0.20,'B':0.20,'C':0.20,'D':0.20,'E':0.20}

    total = sum(bed_config.values()) + 1
    config_F, removed, F_rate = allocate_beds_for_F(
        total, bed_config, arrival_rates, service_rates, relocation_probs,
        urgency_points, target_F_rate=0.95, sim_time=365
    )
    print(f"New config w/ F*: {config_F}, removed from: {removed}, F adm rate={F_rate:.3f}")

    # 3–4) Sensitivity analyses
    bed_configs = {
        'base': bed_config,
        'F_alloc': config_F
    }
    sens = sensitivity_analysis(bed_configs, arrival_rates,
                                service_rates, relocation_probs,
                                urgency_points)

    # 5) Plots
    plot_adm_vs_reloc(base_metrics)
    plot_full_prob(base_metrics)
    # pass urgency_points including F (with 0 if needed)
    plot_penalty_heatmap(sens, {**urgency_points, 'F':0})

    # 6) Generate LaTeX summary & export CSVs
    summary = generate_latex_summary(sens, config_F, best_score=None)
    for name, df in summary['tables'].items():
        df.to_csv(f"metrics_{name}.csv")

if __name__ == "__main__":
    main()
