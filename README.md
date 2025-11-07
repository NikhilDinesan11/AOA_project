# Algorithm Analysis Project

A comprehensive analysis and implementation of two fundamental algorithmic paradigms: **Divide & Conquer** and **Greedy Algorithms**, applied to classic computational problems.

## Table of Contents

- [Overview](#overview)
- [Problems Implemented](#problems-implemented)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Experimental Results](#experimental-results)
- [Contributors](#contributors)

## Overview

This project demonstrates practical implementations of algorithmic design paradigms with rigorous experimental validation. Each implementation includes correctness verification, runtime analysis, and visual comparisons against naive approaches.

## Problems Implemented

### 1. Skyline Problem (Divide & Conquer)

The **Skyline Problem** computes the silhouette formed by a set of overlapping buildings when viewed from a distance. Given a list of buildings represented as `(left, right, height)`, the algorithm outputs key points where the skyline height changes.

**File**: [Divide_and_Conquer.py](Divide_and_Conquer.py)

**Key Features**:
- O(n log n) divide and conquer algorithm
- O(n²) naive algorithm for comparison
- Comprehensive correctness verification
- Runtime analysis with fitted curves
- Visual skyline generation

### 2. Cell Tower Placement (Greedy Algorithm)

The **Cell Tower Placement Problem** determines the minimum number of cell towers needed to cover all towns along a highway, where each tower has a fixed coverage radius.

**File**: [greedy_experiment.py](greedy_experiment.py)

**Key Features**:
- O(n log n) greedy algorithm (sorting + linear scan)
- Handles up to 1 million towns efficiently
- Runtime analysis comparing experimental vs theoretical O(n log n)

## Project Structure

```
AOA_project/
├── Divide_and_Conquer.py    # Skyline problem implementation
├── greedy_experiment.py      # Cell tower placement implementation
├── runtime_graph.png         # D&C runtime analysis with O(n log n) fit
├── comparison_graph.png      # D&C vs Naive algorithm comparison
├── skyline_example.png       # Visual skyline example
├── greedy_plot.png           # Greedy algorithm runtime analysis
└── README.md                 # This file
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AOA_project
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Skyline Problem

Run the complete experimental suite:
```bash
python Divide_and_Conquer.py
```

This will:
1. Verify correctness on 5 test cases
2. Measure runtime for input sizes from 10 to 10,000 buildings
3. Generate three visualization files:
   - `runtime_graph.png` - Runtime vs input size with O(n log n) curve fit
   - `comparison_graph.png` - D&C vs Naive algorithm comparison
   - `skyline_example.png` - Visual representation of a sample skyline

### Cell Tower Placement

Run the greedy algorithm experiment:
```bash
python greedy_experiment.py
```

This will:
1. Test the algorithm on town counts from 10,000 to 1,000,000
2. Measure average execution time over 10 runs per input size
3. Generate `greedy_plot.png` comparing experimental results to theoretical O(n log n)

## Algorithm Details

### Skyline Algorithm (Divide & Conquer)

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

**Approach**:
1. **Base Case**: Single building returns two key points (start and end)
2. **Divide**: Split buildings into two equal halves
3. **Conquer**: Recursively compute skylines for each half
4. **Combine**: Merge two skylines by maintaining maximum height at each position

**Merge Operation**: O(n) time, processes both skylines simultaneously using two pointers

### Cell Tower Placement (Greedy)

**Time Complexity**: O(n log n)
**Space Complexity**: O(1) excluding input storage

**Approach**:
1. Sort town locations: O(n log n)
2. Greedy placement: O(n)
   - Place tower to cover the first uncovered town
   - Position tower at rightmost valid location (town + radius)
   - Skip all towns within coverage range
   - Repeat until all towns covered

**Correctness**: Proven optimal by exchange argument - any optimal solution can be transformed to match greedy solution without increasing tower count

## Experimental Results

### Skyline Problem

Tested on input sizes: 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000 buildings

**Key Findings**:
- D&C algorithm shows strong O(n log n) behavior (high R² fit)
- Significant speedup over naive O(n²) algorithm
- At n=1000: ~30-50× faster than naive approach
- Naive algorithm becomes impractical beyond n=1000

### Cell Tower Placement

Tested on input sizes: 10,000 to 1,000,000 towns (10 data points)

**Key Findings**:
- Experimental results closely follow theoretical O(n log n) curve
- Efficiently handles large-scale inputs (1M towns in seconds)
- Sorting dominates runtime for small inputs
- Linear scan becomes more noticeable at very large scales

## Algorithm Comparison

| Algorithm | Problem | Paradigm | Time Complexity | Space Complexity |
|-----------|---------|----------|-----------------|------------------|
| Skyline | Building silhouette | Divide & Conquer | O(n log n) | O(n) |
| Cell Tower | Coverage optimization | Greedy | O(n log n) | O(1) |
| Naive Skyline | Building silhouette | Brute Force | O(n²) | O(n) |

## Contributors

- Arjun Kaliyath
- Nikhil Dinesan
- darkflames

## License

This project is available for educational purposes.

## References

- The Skyline Problem is a classic divide and conquer problem similar to merge sort
- Cell Tower Placement is a variant of the interval covering problem
- Both problems demonstrate fundamental algorithm design paradigms taught in Analysis of Algorithms courses
