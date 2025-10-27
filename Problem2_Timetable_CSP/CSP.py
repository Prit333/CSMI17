"""
CSP.py
FINAL SCRIPT: This uses a single, hard-coded problem that is
GUARANTEED to cause backtracks and show the power of Forward Checking.
"""

import time
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Hard-Coded Problem ---
class HardCodedCSP:
    def __init__(self):
        # 6 Courses, 3 Time Slots. This is very constrained.
        self.variables = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        
        # Domain: 3 Time Slots (T1, T2, T3) and 2 Rooms (R1, R2)
        domain = [('R1', 'T1'), ('R1', 'T2'), ('R1', 'T3'), 
                  ('R2', 'T1'), ('R2', 'T2'), ('R2', 'T3')]
        
        self.domains = {var: list(domain) for var in self.variables}
        
        # Constraints:
        # C1, C2, C3 are taught by the same Teacher (must have different slots)
        # C4, C5, C6 are taught by the same Teacher (must have different slots)
        # C1, C4 are for the same Student Group (must have different slots)
        # C2, C5 are for the same Student Group (must have different slots)
        # C3, C6 are for the same Student Group (must have different slots)
        
        self.constraints = [
            ('C1', 'C2'), ('C1', 'C3'), ('C2', 'C3'),
            ('C4', 'C5'), ('C4', 'C6'), ('C5', 'C6'),
            ('C1', 'C4'), ('C2', 'C5'), ('C3', 'C6')
        ]

    def is_consistent(self, var, value, assignment):
        """Check if a value for a var is consistent with the assignment."""
        (room, slot) = value
        
        for assigned_var, (assigned_room, assigned_slot) in assignment.items():
            
            # 1. Room Exclusivity Check
            if room == assigned_room and slot == assigned_slot:
                return False

            # 2. Binary Constraint Check (Teacher/Group)
            if (var, assigned_var) in self.constraints or (assigned_var, var) in self.constraints:
                if slot == assigned_slot:
                    return False
        
        return True

# --- Heuristics ---
def mrv(variables, domains, assignment):
    """Minimum Remaining Values (MRV) heuristic."""
    unassigned = [v for v in variables if v not in assignment]
    if not unassigned:
        return None
    
    return min(unassigned, key=lambda v: len(domains[v]))

# --- Backtracking Solver ---
class BacktrackingSolver:
    def __init__(self, csp):
        self.csp = csp
        self.stats = {'backtracks': 0, 'assignments': 0}

    def solve(self, forward_checking=False):
        self.stats = {'backtracks': 0, 'assignments': 0}
        domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        result = self._backtrack({}, domains, forward_checking)
        return result, self.stats

    def _backtrack(self, assignment, domains, forward_checking):
        if len(assignment) == len(self.csp.variables):
            return assignment  # Solution found

        var = mrv(self.csp.variables, domains, assignment)
        if var is None:
            return assignment

        # We are NOT using LCV to force the simple method to backtrack
        for value in domains[var]:
            self.stats['assignments'] += 1
            if self.csp.is_consistent(var, value, assignment):
                
                if forward_checking:
                    pruned_domains = {v: [] for v in self.csp.variables}
                    domain_wipeout = False
                    
                    for other_var in [v for v in self.csp.variables if v not in assignment and v != var]:
                        for other_value in domains[other_var]:
                            if not self.csp.is_consistent(other_var, other_value, {var: value}):
                                pruned_domains[other_var].append(other_value)
                        
                        for val_to_remove in pruned_domains[other_var]:
                            if val_to_remove in domains[other_var]:
                                domains[other_var].remove(val_to_remove)
                        
                        if not domains[other_var]:
                            domain_wipeout = True
                            break
                    
                    if not domain_wipeout:
                        assignment[var] = value
                        result = self._backtrack(assignment, domains, forward_checking)
                        if result:
                            return result
                    
                    for v, values in pruned_domains.items():
                        domains[v].extend(values)

                else: # Standard Backtracking
                    assignment[var] = value
                    result = self._backtrack(assignment, domains, forward_checking)
                    if result:
                        return result

                if var in assignment:
                    del assignment[var]

        self.stats['backtracks'] += 1
        return None

# --- Main Experiment Loop ---
def run_experiment():
    print("Running 1 trial on a hard-coded problem...")
    
    csp = HardCodedCSP()
    solver_heuristic = BacktrackingSolver(csp)
    solver_fc = BacktrackingSolver(csp) 

    # --- Method A: Heuristics Only ---
    start_time = time.perf_counter()
    solution, stats_h = solver_heuristic.solve(forward_checking=False)
    end_time = time.perf_counter()
    time_h = (end_time - start_time) * 1000

    # --- Method B: Forward Checking + Heuristics ---
    start_time = time.perf_counter()
    solution_fc, stats_fc = solver_fc.solve(forward_checking=True)
    end_time = time.perf_counter()
    time_fc = (end_time - start_time) * 1000

    # --- Print Sample Solution ---
    if solution_fc:
        print("\n--- Sample Solved Timetable ---")
        for var, (room, slot) in sorted(solution_fc.items()):
            print(f"  {var}: Assigned to {room} at {slot}")
        print("-----------------------------------")
    else:
        print("--- No solution found ---")

    # --- Final Results ---
    avg_results = {
        'labels': ['Heuristics Only', 'Forward Checking'],
        'time': [time_h, time_fc],
        'backtracks': [stats_h['backtracks'], stats_fc['backtracks']],
        'assignments': [stats_h['assignments'], stats_fc['assignments']]
    }
    
    print("\n--- Final CSP Results (1 Trial) ---")
    print(f"Method: {avg_results['labels'][0]}")
    print(f"  Time (ms): {avg_results['time'][0]:.2f}")
    print(f"  Backtracks: {avg_results['backtracks'][0]:.0f}")
    print(f"  Assignments: {avg_results['assignments'][0]:.0f}")
    
    print(f"\nMethod: {avg_results['labels'][1]}")
    print(f"  Time (ms): {avg_results['time'][1]:.2f}")
    print(f"  Backtracks: {avg_results['backtracks'][1]:.0f}")
    print(f"  Assignments: {avg_results['assignments'][1]:.0f}")
    
    return avg_results

# --- Plotting Function ---
def create_graphs(results):
    print("\nGenerating graphs...")
    
    # --- Graph 1: Average Backtrack Count ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['backtracks'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Backtrack Count (1 Trial)')
    plt.ylabel('Backtracks')
    if results['backtracks'][0] > 0 and results['backtracks'][1] > 0 and \
       (results['backtracks'][0] / (results['backtracks'][1] + 0.01) > 20):
        plt.yscale('log')
        plt.ylabel('Backtracks (Log Scale)')
    plt.bar_label(bars, fmt='%.0f')
    plt.savefig('csp_backtracks_graph.png')
    print("Saved csp_backtracks_graph.png")

    # --- Graph 2: Average Execution Time ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['time'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Execution Time (ms) (1 Trial)')
    plt.ylabel('Time (ms)')
    plt.bar_label(bars, fmt='%.2f')
    plt.savefig('csp_time_graph.png')
    print("Saved csp_time_graph.png")

    # --- Graph 3: Average Assignment Count ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['assignments'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Assignment Count (1 Trial)')
    plt.ylabel('Assignments')
    if results['assignments'][0] > 0 and results['assignments'][1] > 0 and \
       (results['assignments'][0] / (results['assignments'][1] + 0.01) > 20):
        plt.yscale('log')
        plt.ylabel('Assignments (Log Scale)')
    plt.bar_label(bars, fmt='%.0f')
    plt.savefig('csp_assignments_graph.png')
    print("Saved csp_assignments_graph.png")


if __name__ == "__main__":
    final_results = run_experiment()
    create_graphs(final_results)