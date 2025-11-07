import time
import random
import matplotlib.pyplot as plt
import numpy as np


def solve_cell_towers(town_locations, radius):
    """
    Calculates the min towers using the efficient O(n log n) algorithm.
    """
    town_locations.sort() # O(n log n)
    
    n = len(town_locations)
    if n == 0:
        return 0
    
    num_towers = 0
    i = 0
    while i < n: # The whole loop is O(n)
        num_towers += 1
        first_uncovered_town = town_locations[i]
        max_reach = first_uncovered_town + 2 * radius        
        j = i #j will find the first town beyond max_reach
        while j < n and town_locations[j] <= max_reach:
            j += 1
        i = j
        
    return num_towers

def generate_random_towns(n, max_location=10000000):
    """Generates a list of n random, unique town locations within [0, max_location] which are mile markers in the real world."""
    towns = set()
    while len(towns) < n:
        towns.add(random.randint(0, max_location))
    return list(towns)

def run_experiment_greedy():
    """
    Runs the experiment for the O(n log n) algorithm.
    """
    
    # Setting range of n values i.e. the number of towns
    n_values = np.linspace(10000, 1000000, 10).astype(int)
    
    experimental_times = []
    radius = 10
    
    print("N (Towns) | Avg. Time (s)")
    print("-" * 30)

    for n in n_values:
        
        runs_per_n = 10
        total_time = 0
        
        for _ in range(runs_per_n):
            towns = generate_random_towns(n)
            
            # --- Time Greedy ---
            towns_copy = list(towns) 
            start_time = time.perf_counter()
            solve_cell_towers(towns_copy, radius)
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
        
        avg_time = total_time / runs_per_n
        experimental_times.append(avg_time)
        print(f"{n:<10} | {avg_time:.6f}")

    # --- Plotting ---
    
    n_array = np.array(n_values)
    time_array = np.array(experimental_times)

    # Creating an O(n log n) reference curve
    n_log_n = n_array * np.log(n_array)
    c = time_array[-1] / n_log_n[-1] 
    theoretical_curve = c * n_log_n
    
    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_array, time_array, 'bo-', label='Experimental Greedy Time')
    plt.plot(n_array, theoretical_curve, 'r--', label='Theoretical O(n log n) Curve')
    
    plt.xlabel('Number of Towns (n)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Greedy Algorithm Runtime vs. Theoretical O(n log n)')
    plt.legend()
    plt.grid(True)

    plt.ticklabel_format(style='plain', axis='x')
    
    plot_filename = 'greedy_plot.png'
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    
    plt.show()

# --- Main execution ---
if __name__ == "__main__":
    run_experiment_greedy()