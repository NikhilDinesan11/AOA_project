
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def skyline(buildings: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """
    Compute the skyline of a set of buildings using divide and conquer.
    
    Args:
        buildings: List of tuples (left, right, height) representing buildings
                  where left < right and height > 0
    
    Returns:
        List of key points (x, y) representing the skyline.
        Each point indicates where the skyline height changes.
    
    Time Complexity: O(n log n) where n is the number of buildings
    Space Complexity: O(n) for the recursion stack and output
    
    Example:
        >>> buildings = [(1, 5, 11), (2, 7, 6), (3, 9, 13)]
        >>> skyline(buildings)
        [(1, 11), (3, 13), (9, 0)]
    """
    # Base case: no buildings
    if not buildings:
        return []
    
    # Base case: single building
    if len(buildings) == 1:
        left, right, height = buildings[0]
        return [(left, height), (right, 0)]
    
    # Divide: split buildings into two halves
    mid = len(buildings) // 2
    left_buildings = buildings[:mid]
    right_buildings = buildings[mid:]
    
    # Conquer: recursively compute skylines for each half
    left_skyline = skyline(left_buildings)
    right_skyline = skyline(right_buildings)
    
    # Combine: merge the two skylines
    return merge_skylines(left_skyline, right_skyline)


def merge_skylines(left: List[Tuple[int, int]], 
                   right: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge two skylines into a single skyline.
    
    The merge operation scans through both skylines simultaneously,
    maintaining the maximum height at each position.
    
    Args:
        left: Skyline from left half (ordered list of key points)
        right: Skyline from right half (ordered list of key points)
    
    Returns:
        Merged skyline as ordered list of key points
    
    Time Complexity: O(n + m) where n, m are sizes of input skylines
    """
    result = []
    h1, h2 = 0, 0  # Current heights of left and right skylines
    i, j = 0, 0    # Indices for left and right skylines
    
    # Process all key points from both skylines
    while i < len(left) or j < len(right):
        # Determine next x-coordinate to process
        if i >= len(left):
            # Only right skyline has remaining points
            x, h2 = right[j]
            j += 1
        elif j >= len(right):
            # Only left skyline has remaining points
            x, h1 = left[i]
            i += 1
        else:
            # Both skylines have remaining points
            x1, y1 = left[i]
            x2, y2 = right[j]
            
            if x1 < x2:
                # Process point from left skyline
                x, h1 = x1, y1
                i += 1
            elif x2 < x1:
                # Process point from right skyline
                x, h2 = x2, y2
                j += 1
            else:
                # Both skylines have key points at same x-coordinate
                x = x1
                h1, h2 = y1, y2
                i += 1
                j += 1
        
        # The merged skyline height is the maximum of both heights
        max_height = max(h1, h2)
        
        # Add key point only if height changes
        if not result or max_height != result[-1][1]:
            result.append((x, max_height))
    
    return result


def naive_skyline(buildings: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """
    Naive O(n^2) algorithm to compute skyline.
    Used for comparison and validation.
    
    This algorithm checks every x-coordinate where any building starts or ends,
    and computes the maximum height at that position.
    
    Args:
        buildings: List of tuples (left, right, height)
    
    Returns:
        Skyline as list of key points
    
    Time Complexity: O(n^2) where n is the number of buildings
    """
    if not buildings:
        return []
    
    # Collect all critical x-coordinates
    x_coords = set()
    for left, right, height in buildings:
        x_coords.add(left)
        x_coords.add(right)
    
    x_coords = sorted(x_coords)
    
    # For each x-coordinate, find maximum height
    result = []
    for x in x_coords:
        max_height = 0
        for left, right, height in buildings:
            if left <= x < right:
                max_height = max(max_height, height)
        
        # Add key point if height changes
        if not result or max_height != result[-1][1]:
            result.append((x, max_height))
    
    return result


def generate_random_buildings(n: int, 
                              max_x: int = 1000, 
                              max_height: int = 100,
                              seed: int = None) -> List[Tuple[int, int, int]]:
    """
    Generate random buildings for testing.
    
    Args:
        n: Number of buildings to generate
        max_x: Maximum x-coordinate value
        max_height: Maximum building height
        seed: Random seed for reproducibility
    
    Returns:
        List of buildings as (left, right, height) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    buildings = []
    for _ in range(n):
        # Generate random left edge
        left = random.randint(0, max_x - 10)
        # Generate random width (at least 5 units)
        width = random.randint(5, min(50, max_x - left))
        right = left + width
        # Generate random height
        height = random.randint(1, max_height)
        
        buildings.append((left, right, height))
    
    return buildings


def verify_correctness():
    """
    Verify the algorithm produces correct results on known test cases.
    """
    print("=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    
    # Test case 1: Simple non-overlapping buildings
    print("\nTest 1: Non-overlapping buildings")
    buildings = [(0, 2, 10), (3, 5, 15), (6, 8, 12)]
    result = skyline(buildings)
    expected = [(0, 10), (2, 0), (3, 15), (5, 0), (6, 12), (8, 0)]
    print(f"Buildings: {buildings}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ PASS" if result == expected else "✗ FAIL")
    
    # Test case 2: Overlapping buildings
    print("\nTest 2: Overlapping buildings")
    buildings = [(1, 5, 11), (2, 7, 6), (3, 9, 13), (12, 16, 7)]
    result = skyline(buildings)
    naive_result = naive_skyline(buildings)
    print(f"Buildings: {buildings}")
    print(f"D&C Result:   {result}")
    print(f"Naive Result: {naive_result}")
    print(f"✓ PASS" if result == naive_result else "✗ FAIL")
    
    # Test case 3: Buildings with same height
    print("\nTest 3: Buildings with same height")
    buildings = [(0, 3, 5), (2, 6, 5), (4, 8, 5)]
    result = skyline(buildings)
    naive_result = naive_skyline(buildings)
    print(f"Buildings: {buildings}")
    print(f"D&C Result:   {result}")
    print(f"Naive Result: {naive_result}")
    print(f"✓ PASS" if result == naive_result else "✗ FAIL")
    
    # Test case 4: Tall building hiding shorter ones
    print("\nTest 4: Tall building hiding shorter ones")
    buildings = [(1, 5, 10), (2, 4, 5), (2, 4, 3)]
    result = skyline(buildings)
    expected = [(1, 10), (5, 0)]
    print(f"Buildings: {buildings}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ PASS" if result == expected else "✗ FAIL")
    
    # Test case 5: Edge case - single building
    print("\nTest 5: Single building")
    buildings = [(2, 9, 10)]
    result = skyline(buildings)
    expected = [(2, 10), (9, 0)]
    print(f"Buildings: {buildings}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ PASS" if result == expected else "✗ FAIL")
    
    print("\n" + "=" * 60)


def measure_runtime(n_values: List[int], trials: int = 10) -> Tuple[dict, dict]:
    """
    Measure runtime of both algorithms for different input sizes.
    
    Args:
        n_values: List of input sizes to test
        trials: Number of trials per input size (for averaging)
    
    Returns:
        Tuple of (divide_conquer_times, naive_times) dictionaries
    """
    print("\n" + "=" * 60)
    print("RUNTIME ANALYSIS")
    print("=" * 60)
    
    dc_times = {}
    naive_times = {}
    
    for n in n_values:
        print(f"\nTesting n = {n} buildings ({trials} trials)...")
        
        dc_total = 0
        naive_total = 0
        
        for trial in range(trials):
            # Generate random buildings
            buildings = generate_random_buildings(n, seed=trial)
            
            # Measure divide and conquer time
            start = time.perf_counter()
            skyline(buildings)
            end = time.perf_counter()
            dc_total += (end - start)
            
            # Measure naive time (only for smaller inputs)
            if n <= 1000:  # Naive is too slow for large n
                start = time.perf_counter()
                naive_skyline(buildings)
                end = time.perf_counter()
                naive_total += (end - start)
        
        dc_avg = dc_total / trials
        dc_times[n] = dc_avg
        print(f"  Divide & Conquer: {dc_avg*1000:.4f} ms")
        
        if n <= 1000:
            naive_avg = naive_total / trials
            naive_times[n] = naive_avg
            print(f"  Naive Algorithm:  {naive_avg*1000:.4f} ms")
            print(f"  Speedup: {naive_avg/dc_avg:.2f}x")
    
    return dc_times, naive_times


def plot_runtime_analysis(dc_times: dict, output_file: str = "runtime_graph.png"):
    """
    Create a graph showing runtime vs input size with O(n log n) fit.
    
    Args:
        dc_times: Dictionary mapping n -> runtime
        output_file: Output filename for the plot
    """
    # Extract data
    n_values = sorted(dc_times.keys())
    times = [dc_times[n] * 1000 for n in n_values]  # Convert to milliseconds
    
    # Fit O(n log n) curve
    # We find coefficient c such that time ≈ c * n * log(n)
    n_log_n = np.array([n * np.log2(n) for n in n_values])
    
    # Linear regression to find best c
    c = np.sum(np.array(times) * n_log_n) / np.sum(n_log_n ** 2)
    fitted_times = c * n_log_n
    
    # Calculate R-squared
    ss_res = np.sum((np.array(times) - fitted_times) ** 2)
    ss_tot = np.sum((np.array(times) - np.mean(times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(n_values, times, color='blue', s=100, alpha=0.6, label='Experimental Data')
    plt.plot(n_values, fitted_times, color='red', linewidth=2, 
             label=f'Fitted O(n log n) curve (R² = {r_squared:.4f})')
    
    plt.xlabel('Number of Buildings (n)', fontsize=12)
    plt.ylabel('Runtime (milliseconds)', fontsize=12)
    plt.title('Divide & Conquer Skyline Algorithm: Runtime Analysis', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Runtime graph saved to: {output_file}")
    print(f"  R² = {r_squared:.4f} (closer to 1.0 indicates better fit)")
    
    plt.close()


def plot_comparison(dc_times: dict, naive_times: dict, 
                   output_file: str = "comparison_graph.png"):
    """
    Create a comparison graph between divide & conquer and naive algorithms.
    
    Args:
        dc_times: Dictionary mapping n -> runtime for D&C algorithm
        naive_times: Dictionary mapping n -> runtime for naive algorithm
        output_file: Output filename for the plot
    """
    # Extract data (only where both algorithms were measured)
    n_values = sorted([n for n in dc_times.keys() if n in naive_times])
    dc_ms = [dc_times[n] * 1000 for n in n_values]
    naive_ms = [naive_times[n] * 1000 for n in n_values]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, dc_ms, 'o-', color='blue', linewidth=2, 
             markersize=8, label='Divide & Conquer O(n log n)')
    plt.plot(n_values, naive_ms, 's-', color='orange', linewidth=2, 
             markersize=8, label='Naive O(n²)')
    
    plt.xlabel('Number of Buildings (n)', fontsize=12)
    plt.ylabel('Runtime (milliseconds)', fontsize=12)
    plt.title('Algorithm Comparison: Divide & Conquer vs Naive', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Calculate speedup for largest common n
    if n_values:
        max_n = max(n_values)
        speedup = naive_times[max_n] / dc_times[max_n]
        plt.text(0.5, 0.95, f'Speedup at n={max_n}: {speedup:.1f}×', 
                transform=plt.gca().transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison graph saved to: {output_file}")
    
    plt.close()


def visualize_skyline(buildings: List[Tuple[int, int, int]], 
                     output_file: str = "skyline_example.png"):
    """
    Visualize buildings and their computed skyline.
    
    Args:
        buildings: List of buildings to visualize
        output_file: Output filename for the plot
    """
    # Compute skyline
    sky = skyline(buildings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw buildings as rectangles
    for i, (left, right, height) in enumerate(buildings):
        ax.add_patch(plt.Rectangle((left, 0), right - left, height, 
                                   fill=True, alpha=0.3, 
                                   edgecolor='black', linewidth=1.5))
        # Label building
        ax.text((left + right) / 2, height / 2, f'B{i+1}', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw skyline
    x_coords = [point[0] for point in sky]
    y_coords = [point[1] for point in sky]
    
    # Add horizontal segments
    for i in range(len(sky) - 1):
        x1, y1 = sky[i]
        x2, y2 = sky[i + 1]
        ax.plot([x1, x2], [y1, y1], 'r-', linewidth=3, label='Skyline' if i == 0 else '')
        ax.plot([x2, x2], [y1, y2], 'r-', linewidth=3)
    
    # Mark key points
    ax.plot(x_coords, y_coords, 'ro', markersize=8, label='Key Points', zorder=5)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Height', fontsize=12)
    ax.set_title('Skyline Visualization', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(x_coords) - 5, max(x_coords) + 5)
    ax.set_ylim(0, max(y_coords) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Skyline visualization saved to: {output_file}")
    
    plt.close()


def main():
    """
    Main function to run all experiments and generate outputs.
    """
    print("\n" + "=" * 60)
    print("SKYLINE ALGORITHM - EXPERIMENTAL VALIDATION")
    print("=" * 60)
    
    # Step 1: Verify correctness
    verify_correctness()
    
    # Step 2: Measure runtime for different input sizes
    n_values = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    dc_times, naive_times = measure_runtime(n_values, trials=10)
    
    # Step 3: Generate graphs
    print("\n" + "=" * 60)
    print("GENERATING GRAPHS")
    print("=" * 60)
    
    plot_runtime_analysis(dc_times, "runtime_graph.png")
    plot_comparison(dc_times, naive_times, "comparison_graph.png")
    
    # Step 4: Visualize a sample skyline
    print("\n" + "=" * 60)
    print("GENERATING SKYLINE VISUALIZATION")
    print("=" * 60)
    sample_buildings = [(1, 5, 11), (2, 7, 6), (3, 9, 13), (12, 16, 7), 
                        (14, 25, 3), (19, 24, 8)]
    visualize_skyline(sample_buildings, "skyline_example.png")
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - runtime_graph.png")
    print("  - comparison_graph.png")
    print("  - skyline_example.png")
    print("\nThese files are ready to include in your LaTeX report.")


if __name__ == "__main__":
    main()