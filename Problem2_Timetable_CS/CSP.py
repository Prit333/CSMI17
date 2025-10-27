"""
CSP.py
To run: python3 CSP.py
This will also generate graph PNG files.
"""

import random
import time
import matplotlib.pyplot as plt # Import the plotting library
import numpy as np

NUM_TRIALS = 50

# --- CSP Problem Definition ---
class TimetableCSP:
    def __init__(self):
        self.num_courses = 15
        self.num_teachers = 5
        self.num_groups = 4
        self.num_rooms = 5
        self.num_slots = 20  # 5 days * 4 slots

        # Variables: Courses to be scheduled
        self.variables = [f'Course_{i}' for i in range(self.num_courses)]

        # Domains: (Room, Slot) tuples
        self.domains = {var: [] for var in self.variables}
        
        # Constraints
        self.constraints = {}
        self.teacher_of_course = {}
        self.group_of_course = {}

        self.setup_problem()

    def setup_problem(self):
        # 1. Setup Domains and basic unary constraints
        rooms = [(f'R{r}', 'Large' if r < 3 else 'Small') for r in range(self.num_rooms)]
        slots = [f'Slot_{s}' for s in range(self.num_slots)]
        
        course_sizes = {}
        for var in self.variables:
            course_sizes[var] = 'Large' if random.random() > 0.3 else 'Small'

        for var in self.variables:
            for room, capacity in rooms:
                if capacity == 'Large' or course_sizes[var] == 'Small':
                    for slot in slots:
                        self.domains[var].append((room, slot))

        # 2. Setup Binary Constraints (Teachers and Groups)
        courses_per_teacher = self.num_courses // self.num_teachers
        for i in range(self.num_teachers):
            teacher = f'Teacher_{i}'
            for j in range(courses_per_teacher):
                self.teacher_of_course[self.variables[i * courses_per_teacher + j]] = teacher
        
        courses_per_group = self.num_courses // self.num_groups
        for i in range(self.num_groups):
            group = f'Group_{i}'
            for j in range(courses_per_group):
                self.group_of_course[self.variables[i * courses_per_group + j]] = group

    def is_consistent(self, var, value, assignment):
        """Check if a value for a var is consistent with the assignment."""
        (room, slot) = value
        
        for assigned_var, assigned_value in assignment.items():
            (assigned_room, assigned_slot) = assigned_value

            # 1. Room Exclusivity
            if room == assigned_room and slot == assigned_slot:
                return False

            # 2. Teacher Conflict
            if self.teacher_of_course.get(var) == self.teacher_of_course.get(assigned_var) and slot == assigned_slot:
                return False
            
            # 3. Student Group Conflict
            if self.group_of_course.get(var) == self.group_of_course.get(assigned_var) and slot == assigned_slot:
                return False
        
        return True

# --- Heuristics ---
def mrv(variables, domains, assignment):
    """Minimum Remaining Values (MRV) heuristic."""
    unassigned = [v for v in variables if v not in assignment]
    if not unassigned:
        return None
    
    return min(unassigned, key=lambda v: len(domains[v]))

def lcv(var, domains, assignment, csp):
    """Least Constraining Value (LCV) heuristic."""
    def num_conflicts(value):
        conflicts = 0
        for other_var in [v for v in csp.variables if v not in assignment and v != var]:
            # Check only for values still in the *current* domain
            for other_value in domains[other_var]:
                # Simplified check: Assumes 'is_consistent' can check two {var:val} pairs
                # This is a basic LCV, a more complex one would be more precise
                if other_value == value: # Simplified conflict check
                    conflicts += 1
        return conflicts

    return sorted(domains[var], key=num_conflicts)


# --- Backtracking Solver ---
class BacktrackingSolver:
    def __init__(self, csp):
        self.csp = csp
        self.stats = {'backtracks': 0, 'assignments': 0}

    def solve(self, forward_checking=False):
        self.stats = {'backtracks': 0, 'assignments': 0}
        
        # Make a deep copy of domains to modify
        domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        
        result = self._backtrack({}, domains, forward_checking)
        return result, self.stats

    def _backtrack(self, assignment, domains, forward_checking):
        if len(assignment) == len(self.csp.variables):
            return assignment  # Solution found

        var = mrv(self.csp.variables, domains, assignment)
        if var is None:
            return assignment # Should be caught by the check above, but as a fallback

        # Get values ordered by LCV
        ordered_values = lcv(var, domains, assignment, self.csp)
        
        for value in ordered_values:
            self.stats['assignments'] += 1
            if self.csp.is_consistent(var, value, assignment):
                
                # --- Forward Checking Logic ---
                if forward_checking:
                    # Keep track of the domains we are about to prune
                    pruned_domains = {v: [] for v in self.csp.variables}
                    
                    # 1. Look ahead and prune
                    domain_wipeout = False
                    for other_var in [v for v in self.csp.variables if v not in assignment and v != var]:
                        for other_value in domains[other_var]:
                            # Check for conflicts with the new assignment {var: value}
                            if not self.csp.is_consistent(other_var, other_value, {var: value}):
                                pruned_domains[other_var].append(other_value)
                        
                        # Remove the pruned values
                        for val_to_remove in pruned_domains[other_var]:
                            if val_to_remove in domains[other_var]:
                                domains[other_var].remove(val_to_remove)
                        
                        # 2. Check for domain wipeout
                        if not domains[other_var]:
                            domain_wipeout = True
                            break # No point checking others
                    
                    # 3. If no wipeout, recurse
                    if not domain_wipeout:
                        assignment[var] = value
                        result = self._backtrack(assignment, domains, forward_checking)
                        if result:
                            return result
                    
                    # 4. If we backtracked (or had wipeout), restore pruned domains
                    for v, values in pruned_domains.items():
                        domains[v].extend(values)

                # --- Standard Backtracking (No Forward Check) ---
                else:
                    assignment[var] = value
                    result = self._backtrack(assignment, domains, forward_checking)
                    if result:
                        return result

                # If result was None (failure) or we're restoring
                if var in assignment:
                    del assignment[var]

        self.stats['backtracks'] += 1
        return None

# --- Main Experiment Loop ---
def run_experiment():
    results = {
        'heuristic': {'times': [], 'backtracks': [], 'assignments': []},
        'fc': {'times': [], 'backtracks': [], 'assignments': []}
    }
    
    print(f"Running {NUM_TRIALS} trials for CSP...")
    
    for i in range(NUM_TRIALS):
        csp = TimetableCSP()
        solver_heuristic = BacktrackingSolver(csp)
        solver_fc = BacktrackingSolver(csp) # Need a separate instance

        # --- Method A: Heuristics Only ---
        start_time = time.perf_counter()
        solution, stats = solver_heuristic.solve(forward_checking=False)
        end_time = time.perf_counter()
        
        if solution:
            results['heuristic']['times'].append((end_time - start_time) * 1000)
            results['heuristic']['backtracks'].append(stats['backtracks'])
            results['heuristic']['assignments'].append(stats['assignments'])

        # --- Method B: Forward Checking + Heuristics ---
        start_time = time.perf_counter()
        solution_fc, stats_fc = solver_fc.solve(forward_checking=True)
        end_time = time.perf_counter()

        if solution_fc:
            results['fc']['times'].append((end_time - start_time) * 1000)
            results['fc']['backtracks'].append(stats_fc['backtracks'])
            results['fc']['assignments'].append(stats_fc['assignments'])
        
        if not solution or not solution_fc:
            print(f"Trial {i+1} failed to find a solution, retrying...")
            # This is a basic way to handle it; a more robust way would be needed
        else:
            print(f"Trial {i+1}/{NUM_TRIALS} complete.")

    # --- Calculate Averages ---
    avg_results = {
        'labels': ['Heuristics Only', 'Forward Checking'],
        'time': [
            np.mean(results['heuristic']['times']),
            np.mean(results['fc']['times'])
        ],
        'backtracks': [
            np.mean(results['heuristic']['backtracks']),
            np.mean(results['fc']['backtracks'])
        ],
        'assignments': [
            np.mean(results['heuristic']['assignments']),
            np.mean(results['fc']['assignments'])
        ]
    }
    
    print("\n--- Final CSP Results (Averaged) ---")
    print(f"Method: {avg_results['labels'][0]}")
    print(f"  Avg. Time (ms): {avg_results['time'][0]:.2f}")
    print(f"  Avg. Backtracks: {avg_results['backtracks'][0]:.2f}")
    print(f"  Avg. Assignments: {avg_results['assignments'][0]:.2f}")
    
    print(f"\nMethod: {avg_results['labels'][1]}")
    print(f"  Avg. Time (ms): {avg_results['time'][1]:.2f}")
    print(f"  Avg. Backtracks: {avg_results['backtracks'][1]:.2f}")
    print(f"  Avg. Assignments: {avg_results['assignments'][1]:.2f}")
    
    return avg_results

# --- Plotting Function ---
def create_graphs(results):
    print("\nGenerating graphs...")
    
    # --- Graph 1: Average Backtrack Count ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['backtracks'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Average Backtrack Count')
    plt.ylabel('Backtracks')
    # Use a log scale if the difference is huge, otherwise linear
    if results['backtracks'][0] / (results['backtracks'][1] + 1) > 50: # +1 to avoid div by zero
        plt.yscale('log')
        plt.ylabel('Backtracks (Log Scale)')
    
    plt.bar_label(bars, fmt='%.0f')
    plt.savefig('csp_backtracks_graph.png')
    print("Saved csp_backtracks_graph.png")

    # --- Graph 2: Average Execution Time ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['time'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Average Execution Time (ms)')
    plt.ylabel('Time (ms)')
    plt.bar_label(bars, fmt='%.2f')
    plt.savefig('csp_time_graph.png')
    print("Saved csp_time_graph.png")

    # --- Graph 3: Average Assignment Count ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results['labels'], results['assignments'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Average Assignment Count')
    plt.ylabel('Assignments')
    if results['assignments'][0] / (results['assignments'][1] + 1) > 50:
        plt.yscale('log')
        plt.ylabel('Assignments (Log Scale)')
    plt.bar_label(bars, fmt='%.0f')
    plt.savefig('csp_assignments_graph.png')
    print("Saved csp_assignments_graph.png")


if __name__ == "__main__":
    final_results = run_experiment()
    create_graphs(final_results)