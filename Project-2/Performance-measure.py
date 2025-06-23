import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random
from collections import defaultdict
import seaborn as sns
from itertools import product

class HospitalSimulation:
    def __init__(self, bed_capacities, arrival_rates, service_rates, relocation_probs, urgency_points):
        """
        Initialize hospital simulation
        
        Parameters:
        - bed_capacities: dict of ward capacities {ward: capacity}
        - arrival_rates: dict of arrival rates {patient_type: rate per day}
        - service_rates: dict of service rates {patient_type: rate per day}
        - relocation_probs: dict of relocation probabilities {(from_type, to_ward): prob}
        - urgency_points: dict of urgency points {patient_type: points}
        """
        self.bed_capacities = bed_capacities.copy()
        self.arrival_rates = arrival_rates
        self.service_rates = service_rates
        self.relocation_probs = relocation_probs
        self.urgency_points = urgency_points
        
        # Simulation state
        self.current_occupancy = {ward: 0 for ward in bed_capacities.keys()}
        self.time = 0.0
        
        # Statistics tracking
        self.stats = {
            'admissions': defaultdict(int),
            'relocations': defaultdict(int),
            'losses': defaultdict(int),
            'blocked_arrivals': defaultdict(int),
            'occupancy_samples': defaultdict(list),
            'full_ward_events': defaultdict(int)
        }
        
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.current_occupancy = {ward: 0 for ward in self.bed_capacities.keys()}
        self.time = 0.0
        self.stats = {
            'admissions': defaultdict(int),
            'relocations': defaultdict(int),
            'losses': defaultdict(int),
            'blocked_arrivals': defaultdict(int),
            'occupancy_samples': defaultdict(list),
            'full_ward_events': defaultdict(int)
        }
    
    def generate_exponential_time(self, rate):
        """Generate exponential random time"""
        return np.random.exponential(1.0 / rate)
    
    def generate_lognormal_time(self, mean, variance):
        """Generate log-normal random time with specified mean and variance"""
        mu = np.log(mean**2 / np.sqrt(variance + mean**2))
        sigma = np.sqrt(np.log(variance / mean**2 + 1))
        return np.random.lognormal(mu, sigma)
    
    def find_alternative_ward(self, patient_type):
        """Find alternative ward for patient relocation"""
        wards = list(self.bed_capacities.keys())
        probs = []
        available_wards = []
        
        for ward in wards:
            if (patient_type, ward) in self.relocation_probs:
                prob = self.relocation_probs[(patient_type, ward)]
                if prob > 0:
                    probs.append(prob)
                    available_wards.append(ward)
        
        if not available_wards:
            return None
            
        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob == 0:
            return None
            
        probs = [p / total_prob for p in probs]
        
        # Sample ward based on probabilities
        return np.random.choice(available_wards, p=probs)
    
    def simulate_day(self, distribution_type='exponential', variance_multiplier=1.0):
        """Simulate one day of hospital operations"""
        # Generate all events for the day
        events = []
        
        # Generate arrivals for each patient type
        for patient_type in self.arrival_rates:
            rate = self.arrival_rates[patient_type]
            time = 0
            while time < 24:  # 24 hours in a day
                inter_arrival = self.generate_exponential_time(rate / 24)  # Convert to hourly rate
                time += inter_arrival
                if time < 24:
                    events.append(('arrival', time, patient_type))
        
        # Generate departures for currently occupied beds
        for ward in self.current_occupancy:
            for _ in range(self.current_occupancy[ward]):
                # Assume patients are distributed across all types proportionally
                patient_types = list(self.service_rates.keys())
                if patient_types:
                    # Simple assumption: equal distribution of patient types in each ward
                    patient_type = np.random.choice(patient_types)
                    
                    if distribution_type == 'exponential':
                        service_time = self.generate_exponential_time(self.service_rates[patient_type] / 24)
                    else:  # lognormal
                        mean = 1.0 / (self.service_rates[patient_type] / 24)
                        variance = variance_multiplier * (mean ** 2)
                        service_time = self.generate_lognormal_time(mean, variance)
                    
                    if service_time < 24:
                        events.append(('departure', service_time, ward))
        
        # Sort events by time
        events.sort(key=lambda x: x[1])
        
        # Process events
        for event_type, event_time, event_data in events:
            if event_type == 'arrival':
                self.process_arrival(event_data)
            elif event_type == 'departure':
                self.process_departure(event_data)
            
            # Sample occupancy periodically
            if len(self.stats['occupancy_samples'][list(self.current_occupancy.keys())[0]]) < 100:
                for ward in self.current_occupancy:
                    self.stats['occupancy_samples'][ward].append(self.current_occupancy[ward])
    
    def process_arrival(self, patient_type):
        """Process patient arrival"""
        primary_ward = patient_type  # Assume patient type matches primary ward
        
        # Check if primary ward has capacity
        if self.current_occupancy[primary_ward] < self.bed_capacities[primary_ward]:
            # Admit to primary ward
            self.current_occupancy[primary_ward] += 1
            self.stats['admissions'][primary_ward] += 1
        else:
            # Primary ward is full
            self.stats['blocked_arrivals'][primary_ward] += 1
            self.stats['full_ward_events'][primary_ward] += 1
            
            # Try to find alternative ward
            alternative_ward = self.find_alternative_ward(patient_type)
            
            if alternative_ward and self.current_occupancy[alternative_ward] < self.bed_capacities[alternative_ward]:
                # Admit to alternative ward
                self.current_occupancy[alternative_ward] += 1
                self.stats['admissions'][alternative_ward] += 1
                self.stats['relocations'][patient_type] += 1
            else:
                # Patient is lost from system
                self.stats['losses'][patient_type] += 1
    
    def process_departure(self, ward):
        """Process patient departure"""
        if self.current_occupancy[ward] > 0:
            self.current_occupancy[ward] -= 1
    
    def run_simulation(self, num_days=365, distribution_type='exponential', variance_multiplier=1.0):
        """Run simulation for specified number of days"""
        self.reset_simulation()
        
        for day in range(num_days):
            self.simulate_day(distribution_type, variance_multiplier)
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        metrics = {}
        
        for ward in self.bed_capacities:
            total_arrivals = self.stats['admissions'][ward] + self.stats['blocked_arrivals'].get(ward, 0)
            
            # Probability that all beds are occupied on arrival
            if self.stats['full_ward_events'][ward] > 0:
                prob_full = self.stats['full_ward_events'][ward] / max(1, total_arrivals)
            else:
                prob_full = 0.0
            
            metrics[ward] = {
                'prob_all_beds_occupied': prob_full,
                'expected_admissions': self.stats['admissions'][ward],
                'expected_relocations_from': sum(self.stats['relocations'][pt] for pt in self.stats['relocations'] 
                                               if pt == ward),
                'expected_relocations_to': self.stats['admissions'][ward] - 
                                         sum(1 for pt in self.arrival_rates.keys() if pt == ward) * 
                                         self.stats['admissions'][ward] / max(1, sum(self.stats['admissions'].values())),
                'losses': sum(self.stats['losses'][pt] for pt in self.stats['losses'] if pt == ward),
                'avg_occupancy': np.mean(self.stats['occupancy_samples'][ward]) if self.stats['occupancy_samples'][ward] else 0
            }
        
        return metrics

def setup_base_parameters():
    """Setup base parameters from the problem statement"""
    # Ward capacities (excluding F*)
    bed_capacities = {'A': 55, 'B': 40, 'C': 30, 'D': 20, 'E': 20}
    
    # Arrival rates (patients per day)
    arrival_rates = {'A': 14.5, 'B': 11.0, 'C': 8.0, 'D': 6.5, 'E': 5.0, 'F': 13.0}
    
    # Service rates (departures per day = 1/mean_length_of_stay)
    service_rates = {'A': 1/2.9, 'B': 1/4.0, 'C': 1/4.5, 'D': 1/1.4, 'E': 1/3.9, 'F': 1/2.2}
    
    # Urgency points
    urgency_points = {'A': 7, 'B': 5, 'C': 2, 'D': 10, 'E': 5, 'F': 0}
    
    # Relocation probabilities
    relocation_probs = {
        ('A', 'B'): 0.05, ('A', 'C'): 0.10, ('A', 'D'): 0.05, ('A', 'E'): 0.80,
        ('B', 'A'): 0.20, ('B', 'C'): 0.50, ('B', 'D'): 0.15, ('B', 'E'): 0.15,
        ('C', 'A'): 0.30, ('C', 'B'): 0.20, ('C', 'D'): 0.20, ('C', 'E'): 0.30,
        ('D', 'A'): 0.35, ('D', 'B'): 0.30, ('D', 'C'): 0.05, ('D', 'E'): 0.30,
        ('E', 'A'): 0.20, ('E', 'B'): 0.10, ('E', 'C'): 0.60, ('E', 'D'): 0.10,
        ('F', 'A'): 0.20, ('F', 'B'): 0.20, ('F', 'C'): 0.20, ('F', 'D'): 0.20, ('F', 'E'): 0.20
    }
    
    return bed_capacities, arrival_rates, service_rates, relocation_probs, urgency_points

def task1_base_simulation():
    """Task 1: Build and run base simulation model"""
    print("=== TASK 1: Base Simulation Model ===")
    
    bed_capacities, arrival_rates, service_rates, relocation_probs, urgency_points = setup_base_parameters()
    
    # Run simulation without Ward F*
    base_arrival_rates = {k: v for k, v in arrival_rates.items() if k != 'F'}
    base_service_rates = {k: v for k, v in service_rates.items() if k != 'F'}
    
    sim = HospitalSimulation(bed_capacities, base_arrival_rates, base_service_rates, 
                           relocation_probs, urgency_points)
    
    print("Running base simulation...")
    metrics = sim.run_simulation(num_days=365)
    
    # Display results
    print("\nBase System Performance Metrics:")
    print("-" * 60)
    for ward in sorted(metrics.keys()):
        print(f"Ward {ward}:")
        print(f"  Probability all beds occupied: {metrics[ward]['prob_all_beds_occupied']:.3f}")
        print(f"  Expected admissions: {metrics[ward]['expected_admissions']}")
        print(f"  Expected relocations: {metrics[ward]['expected_relocations_from']}")
        print(f"  Average occupancy: {metrics[ward]['avg_occupancy']:.1f}")
        print()
    
    return sim, metrics

def task2_optimize_ward_f():
    """Task 2: Create Ward F* and optimize bed allocation"""
    print("=== TASK 2: Ward F* Optimization ===")
    
    bed_capacities, arrival_rates, service_rates, relocation_probs, urgency_points = setup_base_parameters()
    
    # Total beds available for reallocation
    total_beds = sum(bed_capacities.values())
    print(f"Total beds available: {total_beds}")
    
    best_allocation = None
    best_score = float('inf')
    best_f_admission_rate = 0
    
    # Try different allocations for Ward F*
    print("Optimizing Ward F* allocation...")
    
    for f_beds in range(5, 31):  # Try 5 to 30 beds for Ward F*
        # Calculate remaining beds to distribute
        remaining_beds = total_beds - f_beds
        
        # Proportional reduction from other wards
        reduction_factor = remaining_beds / total_beds
        
        new_capacities = bed_capacities.copy()
        for ward in ['A', 'B', 'C', 'D', 'E']:
            new_capacities[ward] = max(1, int(bed_capacities[ward] * reduction_factor))
        new_capacities['F'] = f_beds
        
        # Ensure total doesn't exceed original
        actual_total = sum(new_capacities.values())
        if actual_total > total_beds:
            # Reduce F* beds to match total
            new_capacities['F'] = f_beds - (actual_total - total_beds)
        
        # Run simulation
        sim = HospitalSimulation(new_capacities, arrival_rates, service_rates, 
                               relocation_probs, urgency_points)
        metrics = sim.run_simulation(num_days=365)
        
        # Calculate F* admission rate
        f_admissions = metrics.get('F', {}).get('expected_admissions', 0)
        f_arrival_rate = arrival_rates['F'] * 365  # Total F arrivals per year
        f_admission_rate = f_admissions / f_arrival_rate if f_arrival_rate > 0 else 0
        
        # Calculate weighted penalty score using urgency points
        penalty_score = 0
        for ward in ['A', 'B', 'C', 'D', 'E']:
            if ward in metrics:
                relocations = metrics[ward].get('expected_relocations_from', 0)
                urgency = urgency_points[ward]
                penalty_score += relocations * urgency
        
        # Check if F* admission rate meets requirement
        if f_admission_rate >= 0.95:
            if penalty_score < best_score:
                best_score = penalty_score
                best_allocation = new_capacities.copy()
                best_f_admission_rate = f_admission_rate
    
    if best_allocation:
        print(f"\nOptimal allocation found:")
        print(f"Ward F* admission rate: {best_f_admission_rate:.3f}")
        print(f"Penalty score: {best_score:.2f}")
        print(f"Bed allocation: {best_allocation}")
        
        # Run final simulation with optimal allocation
        sim = HospitalSimulation(best_allocation, arrival_rates, service_rates, 
                               relocation_probs, urgency_points)
        metrics = sim.run_simulation(num_days=365)
        
        print("\nOptimal System Performance Metrics:")
        print("-" * 60)
        for ward in sorted(metrics.keys()):
            print(f"Ward {ward}:")
            print(f"  Bed capacity: {best_allocation[ward]}")
            print(f"  Probability all beds occupied: {metrics[ward]['prob_all_beds_occupied']:.3f}")
            print(f"  Expected admissions: {metrics[ward]['expected_admissions']}")
            print(f"  Average occupancy: {metrics[ward]['avg_occupancy']:.1f}")
            print()
        
        return best_allocation, metrics
    else:
        print("No allocation found that meets the 95% admission rate requirement for Ward F*")
        return None, None

def task3_assess_implications(base_metrics, optimal_allocation, optimal_metrics):
    """Task 3: Assess implications of creating Ward F*"""
    print("=== TASK 3: Implications Assessment ===")
    
    if optimal_allocation is None:
        print("Cannot assess implications - no optimal allocation found")
        return
    
    print("Comparison: Before vs After Ward F* Creation")
    print("-" * 60)
    
    # Compare key metrics
    total_admissions_before = sum(base_metrics[ward]['expected_admissions'] for ward in base_metrics)
    total_admissions_after = sum(optimal_metrics[ward]['expected_admissions'] for ward in optimal_metrics)
    
    print(f"Total admissions - Before: {total_admissions_before}, After: {total_admissions_after}")
    print(f"Change in total admissions: {total_admissions_after - total_admissions_before}")
    
    print("\nWard-by-ward comparison:")
    for ward in ['A', 'B', 'C', 'D', 'E']:
        if ward in base_metrics and ward in optimal_metrics:
            before_occ = base_metrics[ward]['avg_occupancy']
            after_occ = optimal_metrics[ward]['avg_occupancy']
            print(f"Ward {ward} - Occupancy change: {before_occ:.1f} → {after_occ:.1f} ({after_occ-before_occ:+.1f})")
    
    print(f"\nNew Ward F*:")
    if 'F' in optimal_metrics:
        print(f"  Capacity: {optimal_allocation['F']}")
        print(f"  Expected admissions: {optimal_metrics['F']['expected_admissions']}")
        print(f"  Average occupancy: {optimal_metrics['F']['avg_occupancy']:.1f}")

def sensitivity_analysis():
    """Sensitivity analysis for different distributions and bed allocations"""
    print("=== SENSITIVITY ANALYSIS ===")
    
    bed_capacities, arrival_rates, service_rates, relocation_probs, urgency_points = setup_base_parameters()
    
    # Test log-normal distribution with different variances
    print("Testing log-normal distribution sensitivity...")
    
    base_sim = HospitalSimulation(bed_capacities, 
                                {k: v for k, v in arrival_rates.items() if k != 'F'}, 
                                {k: v for k, v in service_rates.items() if k != 'F'}, 
                                relocation_probs, urgency_points)
    
    variance_multipliers = [1.0, 2.0, 3.0, 4.0]
    
    print("\nLength-of-stay distribution sensitivity:")
    print("-" * 50)
    
    for var_mult in variance_multipliers:
        print(f"\nVariance multiplier: {var_mult}")
        
        # Exponential baseline
        exp_metrics = base_sim.run_simulation(num_days=100, distribution_type='exponential')
        
        # Log-normal with variance
        ln_metrics = base_sim.run_simulation(num_days=100, distribution_type='lognormal', 
                                           variance_multiplier=var_mult)
        
        print("Ward | Exp Admissions | LN Admissions | Difference")
        for ward in sorted(exp_metrics.keys()):
            exp_adm = exp_metrics[ward]['expected_admissions']
            ln_adm = ln_metrics[ward]['expected_admissions']
            diff = ln_adm - exp_adm
            print(f"  {ward}  |     {exp_adm:6.0f}    |    {ln_adm:6.0f}    |   {diff:+6.0f}")
    
    # Test different total bed counts
    print("\n\nTotal bed count sensitivity:")
    print("-" * 40)
    
    bed_counts = [150, 160, 165, 170, 180, 190]
    
    for total_beds in bed_counts:
        # Scale bed capacities proportionally
        scale_factor = total_beds / sum(bed_capacities.values())
        scaled_capacities = {ward: max(1, int(cap * scale_factor)) 
                           for ward, cap in bed_capacities.items()}
        
        # Adjust to match exact total
        actual_total = sum(scaled_capacities.values())
        if actual_total != total_beds:
            # Add/remove from largest ward
            largest_ward = max(scaled_capacities.keys(), key=lambda x: scaled_capacities[x])
            scaled_capacities[largest_ward] += (total_beds - actual_total)
        
        sim = HospitalSimulation(scaled_capacities,
                               {k: v for k, v in arrival_rates.items() if k != 'F'}, 
                               {k: v for k, v in service_rates.items() if k != 'F'}, 
                               relocation_probs, urgency_points)
        
        metrics = sim.run_simulation(num_days=100)
        
        total_admissions = sum(metrics[ward]['expected_admissions'] for ward in metrics)
        avg_occupancy = sum(metrics[ward]['avg_occupancy'] for ward in metrics)
        
        print(f"Total beds: {total_beds} | Admissions: {total_admissions:6.0f} | Avg occupancy: {avg_occupancy:6.1f}")

def main():
    """Main function to run all tasks"""
    print("Hospital Bed Allocation Simulation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Task 1: Base simulation
    base_sim, base_metrics = task1_base_simulation()
    
    # Task 2: Optimization
    optimal_allocation, optimal_metrics = task2_optimize_ward_f()
    
    # Task 3: Assessment
    if optimal_allocation and optimal_metrics:
        task3_assess_implications(base_metrics, optimal_allocation, optimal_metrics)
    
    # Sensitivity Analysis
    sensitivity_analysis()
    
    print("\n" + "=" * 50)
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()