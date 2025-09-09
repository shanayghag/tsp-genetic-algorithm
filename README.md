# TSP Genetic Algorithm

An advanced Genetic Algorithm implementation for solving the Traveling Salesman Problem. Features intelligent k-nearest neighbor population initialization, TSP-optimized 2-opt mutation, multi-tier breeding strategies, and adaptive parameter scaling

## 🚀 Features

- **Intelligent Population Initialization**: K-nearest neighbor heuristic for high-quality starting solutions
- **TSP-Specific Optimization**: 2-opt inversion mutation eliminates edge crossings
- **Multi-Tier Breeding**: Elite preservation + elite breeding + general population crossover
- **Adaptive Parameters**: Problem size-aware parameter tuning
- **3D Coordinate Support**: Handles 3D Euclidean distance calculations
- **Performance Monitoring**: Real-time progress tracking and execution timing

## 📋 Problem Statement

The **Traveling Salesman Problem (TSP)** is a classic optimization problem where a salesman must visit all cities exactly once and return to the starting city, minimizing the total travel distance

### Mathematical Formulation
- **Input**: Set of cities with coordinates `{C₁, C₂, ..., Cₙ}`
- **Objective**: Find the shortest Hamiltonian cycle
- **Constraint**: Visit each city exactly once
- **Goal**: Minimize total distance: `∑ᵢ₌₁ⁿ d(Cᵢ, Cᵢ₊₁) + d(Cₙ, C₁)`

### Complexity
- **NP-Hard** problem with `O(n!)` possible solutions
- Exact algorithms become infeasible for large instances (n > 20)
- Heuristic approaches like genetic algorithms provide near-optimal solutions

## 🧬 Algorithm Overview

### Core Genetic Algorithm Components

1. **Population Representation**: Permutation encoding (city sequence)
2. **Fitness Function**: `fitness = 1 / total_tour_distance`
3. **Selection**: Elitism + Roulette Wheel Selection
4. **Crossover**: Two-point crossover with permutation repair
5. **Mutation**: 2-opt segment inversion
6. **Replacement**: Multi-population strategy

## 🔧 Key Algorithmic Innovations

### 1. K-Nearest Neighbor Initialization
```python
# Instead of random initialization
population = random_permutations(cities)

# Use intelligent KNN-based seeding
population = knn_population_initialization(k, coordinates, population_size)
```

**Benefits:**
- Generates high-quality initial solutions
- Reduces convergence time
- Provides better starting point than random initialization

### 2. TSP-Optimized 2-Opt Mutation
```python
# Traditional swap mutation
swap(chromosome[i], chromosome[j])

# 2-opt inversion mutation
chromosome[start:end] = chromosome[start:end][::-1]
```

**Benefits:**
- Eliminates edge crossings in tours
- Maintains tour connectivity
- More effective for TSP than random swaps

### 3. Multi-Tier Breeding Strategy

**Tier 1: Elite Preservation**
- Preserve top `elitism_rate * population_size` individuals

**Tier 2: Elite Breeding**
- Crossover among elite individuals only
- Generates high-quality offspring

**Tier 3: General Population Breeding**
- Crossover with immediate mutation
- Maintains population diversity

### 4. Adaptive Parameter Scaling
```python
if num_cities <= 20:
    k = num_cities - 1      # Use all neighbors
elif num_cities <= 50:
    k = 20                  # Medium problems
else:
    k = 40                  # Large problems
```

## 🛠️ Implementation Details

### Class Structure

```
Utils                           # File I/O and distance calculations
├── get_euclidean_distance()    # 3D Euclidean distance
├── read_input_file()           # Parse input format
└── write_output_file()         # Generate solution output

CrossoverMethods               # Genetic operators
├── cycle_crossover()          # Cycle crossover implementation
└── two_point_crossover()      # Two-point with repair

PopulationInitializationMethods # Smart initialization
├── get_k_nearest_neighbors()   # Efficient KNN computation
├── generate_knn_path()         # KNN-based tour construction
└── knn_population_initialization()

GeneticAlgorithm              # Main GA engine
├── get_fitness()             # Fitness evaluation
├── rank_population()         # Population ranking
├── create_mating_pool()      # Selection mechanism
├── crossover_operations()    # Multi-tier breeding
├── mutate()                  # 2-opt mutation
└── get_next_generation()     # Evolution step
```

### Algorithm Flow

1. **Initialize Population**
   ```python
   population = knn_population_initialization(k, coordinates, population_size)
   ```

2. **Evolution Loop**
   ```python
   for generation in range(num_generations):
       # Selection
       ranked_pop, fitness_sum = rank_population(population)
       mating_pool = create_mating_pool(ranked_pop, fitness_sum, elitism_rate)
       
       # Crossover (Multi-tier)
       offspring = multi_tier_crossover(mating_pool, elitism_rate)
       
       # Mutation
       mutated_pop = mutate_population(offspring)
       
       # Replacement
       population = combine_populations(offspring, mutated_pop)
   ```

3. **Output Best Solution**
   ```python
   best_tour = get_best_individual(population)
   write_output_file(best_tour)
   ```

## 📁 File Structure

```
├── genetic_algorithm.py       # Main implementation
├── inputs/                    # Input test cases
│   ├── input1.txt
│   ├── input5.txt
│   └── ...
├── output.txt                # Solution output
└── README.md                 # This file
```

## 📝 Input Format

```
n
x1 y1 z1
x2 y2 z2
...
xn yn zn
```

- **Line 1**: Number of cities `n`
- **Lines 2 to n+1**: 3D coordinates of each city

## 📤 Output Format

```
x1 y1 z1
x2 y2 z2
...
xn yn zn
x1 y1 z1
```

- Optimal tour sequence (coordinates of cities in visit order)
- Returns to starting city to complete the cycle

## 🚀 Usage

### Basic Usage
```python
# Use default parameters and input file
python genetic_algorithm.py
```

### Custom Input File
```python
# Modify run function call
run("./inputs/input10.txt")
```

### Programmatic Usage
```python
from genetic_algorithm import solve_tsp, Utils

# Load problem
coordinates_2_cities, coordinates = Utils.map_coordinates_to_city_nums("input.txt")

# Solve with custom parameters
solve_tsp(
    coordinates_2_cities=coordinates_2_cities,
    coordinates=coordinates,
    population_size=200,
    elitism_rate=0.1,
    num_generations=1000
)
```

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of individuals in population |
| `elitism_rate` | 0.2 | Fraction of best individuals preserved |
| `num_generations` | 500 | Number of evolution iterations |
| `k` | Adaptive | Number of nearest neighbors (auto-scaled) |

### Parameter Tuning Guidelines

**Population Size:**
- Small problems (n ≤ 20): 50-100
- Medium problems (20 < n ≤ 50): 100-200  
- Large problems (n > 50): 200-500

**Elitism Rate:**
- Conservative: 0.1-0.15 (more exploration)
- Balanced: 0.2-0.25 (recommended)
- Aggressive: 0.3+ (faster convergence, less diversity)

**Generations:**
- Quick test: 100-250
- Standard: 500-1000
- Thorough: 1000-2000

## 📊 Performance

### Complexity Analysis
- **Time Complexity**: `O(g × p × n²)` where:
  - `g` = generations
  - `p` = population size  
  - `n` = number of cities
- **Space Complexity**: `O(p × n)`

## 🔬 Algorithm Analysis

### Strengths
- **Fast Convergence**: KNN initialization provides excellent starting point
- **TSP-Optimized**: 2-opt mutation specifically designed for tour problems
- **Scalable**: Adaptive parameters handle various problem sizes
- **Robust**: Multi-population strategy maintains diversity

### Limitations
- **Local Optima**: May converge to local minima for very large instances
- **Parameter Sensitivity**: Performance depends on parameter tuning
- **Memory Usage**: Stores multiple populations simultaneously

### Possible Improvements
- **Hybrid Approach**: Combine with local search (Lin-Kernighan)
- **Parallel Processing**: Distribute fitness evaluations
- **Advanced Operators**: Implement Order Crossover (OX) or PMX
- **Adaptive Mutation**: Dynamic mutation rates based on diversity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📚 References

1. Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
2. Lawler, E.L. et al. (1985). *The Traveling Salesman Problem*
3. Lin, S. & Kernighan, B.W. (1973). "An Effective Heuristic Algorithm for the TSP"
4. Reinelt, G. (1994). *The Traveling Salesman Problem: Computational Solutions*

---

*For questions or issues, please open a GitHub issue or contact [shanayghag200@gmail.com]*
