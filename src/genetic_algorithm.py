import random
import time
from itertools import permutations
import copy

import numpy as np
from scipy.spatial.distance import pdist, squareform


class Utils:
    """Utility class for distance calculations, file I/O operations, and helper functions."""
    
    @staticmethod
    def get_euclidean_distance(p1, p2):
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            p1 (list): First point coordinates [x, y, z]
            p2 (list): Second point coordinates [x, y, z]
            
        Returns:
            float: Euclidean distance between the two points
        """
        return (((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2) + ((p1[2] - p2[2])**2))**0.5

    @staticmethod
    def read_input_file(filepath):
        """
        Read TSP problem input from a specific input file.
        Expected format:
        - First line: number of cities
        - Subsequent lines: x y z coordinates of each city
        
        Returns:
            tuple: (number of cities, list of city coordinates)
        """
        with open(filepath, "r") as f:
            lines = f.readlines()
            num_cities = int(lines[0])
            city_coordinates = []
            for i in range(1, len(lines)):
                temp = lines[i].replace("\n", "").split(" ")
                city_coordinates.append(list(int(j) for j in temp))
            return num_cities, city_coordinates

    @staticmethod
    def write_output_file(chromosome, cities_2_coordinate):
        """
        Write the best TSP solution to output file.
        Outputs the complete tour including return to starting city.
        
        Args:
            chromosome (list): Best tour sequence (list of city indices)
            cities_2_coordinate (dict): Mapping from city index to coordinates
        """
        with open("./output.txt", "w") as f:
            temp_str = ""
            for gene in chromosome:
                gene_ = cities_2_coordinate[gene]
                temp_str += " ".join(str(g) for g in gene_)
                temp_str += "\n"
            # Return to starting city to complete the tour
            gene_ = cities_2_coordinate[chromosome[0]]
            temp_str += " ".join(str(g) for g in gene_)
            f.writelines(temp_str)

    @staticmethod
    def map_coordinates_to_city_nums(filepath):
        """
        Create mapping between city indices and their coordinates.
        
        Returns:
            tuple: (dict mapping city index to coordinates, list of all coordinates)
        """
        num_cities, city_coordinates = Utils.read_input_file(filepath)
        coordinates_2_cities = {}
        for i in range(num_cities):
            coordinates_2_cities[i] = city_coordinates[i]
        return coordinates_2_cities, city_coordinates


class CrossoverMethods:
    """Contains different crossover methods for genetic algorithm operations."""
    
    @staticmethod
    def cycle_crossover(parent1, parent2):
        """
        Perform cycle crossover between two parent chromosomes.
        Cycle crossover preserves the position relationship between cities from both parents.
        
        Args:
            parent1 (list): First parent chromosome
            parent2 (list): Second parent chromosome
            
        Returns:
            tuple: Two child chromosomes (child1, child2)
        """
        parent1_lookup = {gene: i for i, gene in enumerate(parent1)}
        cycles = [-1] * len(parent1)
        start_positions = [i for i, gene in enumerate(cycles) if gene < 0]
        
        for c_num, pos in enumerate(start_positions):
            while cycles[pos] < 0:
                cycles[pos] = c_num
                pos = parent1_lookup[parent2[pos]]   
                
        child1 = [parent1[i] if gene % 2 else parent2[i] for i, gene in enumerate(cycles)]
        child2 = [parent2[i] if gene % 2 else parent1[i] for i, gene in enumerate(cycles)]
        return child1, child2

    @staticmethod
    def two_point_crossover(parent1, parent2, start_idx, end_idx):
        """
        Perform two-point crossover between parent chromosomes.
        Takes a segment from parent1 and fills remaining positions with parent2's genes
        while maintaining the permutation property (no duplicate cities).
        
        Args:
            parent1 (list): First parent chromosome
            parent2 (list): Second parent chromosome
            start_idx (int): Starting index for crossover segment
            end_idx (int): Ending index for crossover segment
            
        Returns:
            list: Child chromosome
        """
        child = [-1] * len(parent1)
        
        # Copy segment from parent1
        for i in range(start_idx, end_idx):
            child[i] = parent1[i]
            
        # Fill remaining positions with genes from parent2
        for j in range(len(child)):
            if child[j] == -1:
                if parent2[j] not in child:
                    child[j] = parent2[j]
                else:
                    # Find missing gene to maintain permutation property
                    set_parent2 = set(parent1)
                    set_child = set(child)
                    difference = set_parent2.difference(set_child)
                    child[j] = list(difference)[0]
        return child


class PopulationInitializationMethods:
    """Methods for intelligent population initialization using k-nearest neighbors heuristic."""
    
    @staticmethod
    def get_k_nearest_neighbors(k, X, method="euclidean", return_distances=False):
        """
        Find k nearest neighbors for each point in the dataset using efficient algorithms.
        
        Args:
            k (int): Number of nearest neighbors to find
            X (np.array): Array of point coordinates
            method (str): Distance metric to use ('euclidean' for optimized version)
            return_distances (bool): Whether to return distances along with indices
            
        Returns:
            np.array or tuple: Indices of k nearest neighbors (and distances if requested)
        """
        # Calculate pairwise distance matrix
        D_cond = pdist(X, method)
        D = squareform(D_cond)

        if method == "euclidean":
            # Optimized euclidean distance calculation using matrix operations
            XX = np.einsum('ij, ij ->i', X, X)
            # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
            D_sq = XX[:, None] + XX - 2 * X.dot(X.T)

            # Find indices of the k nearest neighbors
            if k == 1:
                k_i = D_sq.argmin(axis=1)
            else:
                k_i = D_sq.argpartition(kth=k, axis=-1)
                k_i = k_i[:, :k]

            if return_distances:
                k_d = np.sqrt(np.take_along_axis(D_sq, k_i, axis=1))
                k_d = np.nan_to_num(k_d)
                sorted_idx = k_d.argsort(axis=1)
                k_i_sorted = np.take_along_axis(k_i, sorted_idx, axis=1)
                k_d_sorted = np.take_along_axis(k_d, sorted_idx, axis=1)
                return k_i_sorted, k_d_sorted

            # Sort by distance without computing sqrt (more efficient)
            k_d_sq = np.take_along_axis(D_sq, k_i, axis=1)
            sorted_idx = k_d_sq.argsort(axis=1)
            k_i_sorted = np.take_along_axis(k_i, sorted_idx, axis=1)
            return k_i_sorted

        else:
            # Standard implementation for other distance metrics
            if k == 1:
                k_i = D.argmin(axis=1)
            else:
                k_i = D.argpartition(kth=k, axis=-1)
                k_i = k_i[:, :k]
                k_d = np.take_along_axis(D, k_i, axis=1)
                sorted_idx = k_d.argsort(axis=1)
                k_i_sorted = np.take_along_axis(k_i, sorted_idx, axis=1)
                k_d_sorted = np.take_along_axis(k_d, sorted_idx, axis=1)
                
                if return_distances:
                    return k_i_sorted, k_d_sorted
                else:
                    return k_i_sorted

    @staticmethod
    def generate_knn_path(knns, start_city, city_set):
        """
        Generate a TSP path using k-nearest neighbor heuristic starting from a specific city.
        
        Args:
            knns (list): k-nearest neighbors for each city
            start_city (int): Starting city index
            city_set (set): Set of all city indices
            
        Returns:
            list: Generated path as sequence of city indices
        """
        current_city = start_city
        path = [current_city]
        
        for i in range(len(knns) - 1):
            city_flag = False
            neighbors = knns[current_city][1:]  # Skip self (first neighbor is always self)
            
            # Try to find unvisited neighbor
            for j in range(len(neighbors)):
                if neighbors[j] not in path:
                    current_city = neighbors[j]
                    path.append(current_city)
                    city_flag = True
                    break
                    
            # If no unvisited neighbors found, pick any unvisited city
            if not city_flag:
                path_set = set(path)
                difference = city_set.difference(path_set)
                if not difference:
                    return path
                else:
                    current_city = list(difference)[0]
                    path.append(current_city)
        return path

    @staticmethod
    def knn_population_initialization(k, coordinates, coordinates_2_cities, population_size):
        """
        Initialize population using k-nearest neighbor heuristic for better starting solutions.
        This method creates high-quality initial solutions and then ranks them.
        
        Args:
            k (int): Number of nearest neighbors to consider
            coordinates (list): List of city coordinates
            coordinates_2_cities (dict): Mapping from city index to coordinates
            population_size (int): Desired population size
            
        Returns:
            list: Initial population of chromosomes, ranked by fitness
        """
        city_keys = list(coordinates_2_cities.keys())
        
        # For very small problems, generate all permutations
        if len(city_keys) <= 5:
            population = list(permutations(city_keys))
            population = [list(population[i]) for i in range(len(population))]
        else:
            X = np.array(coordinates)
            knns = PopulationInitializationMethods.get_k_nearest_neighbors(k, X)
            knns = knns.tolist()
            population = []
            
            # Generate one high-quality path starting from each city using KNN
            for i in range(len(city_keys)):
                chromosome = PopulationInitializationMethods.generate_knn_path(knns, i, set(city_keys))
                population.append(chromosome)
            
            # Fill remaining population with random chromosomes
            i = 0
            while i < (population_size - len(city_keys)):
                temp = random.sample(city_keys, len(city_keys))
                if temp not in population:
                    population.append(temp)
                    i += 1
                else:
                    continue
            
            # Rank population by fitness and return sorted by quality
            ranked_population, _ = GeneticAlgorithm.rank_population(population, coordinates_2_cities)
            population = [ranked_population[i][1] for i in range(len(ranked_population))]
        
        return population[:population_size]

    @staticmethod
    def create_initial_population(coordinates_2_cities, population_size):
        """
        Create initial population with random chromosomes (fallback method).
        
        Args:
            coordinates_2_cities (dict): Mapping from city index to coordinates
            population_size (int): Size of population to create
            
        Returns:
            list: Initial population of random chromosomes
        """
        city_keys = list(coordinates_2_cities.keys())
        
        if len(city_keys) <= 5:
            # Generate all permutations for very small problems
            population = list(permutations(city_keys))
            population = [list(population[i]) for i in range(len(population))]
        else:
            population = []
            i = 0
            while i < population_size:
                temp = random.sample(city_keys, len(city_keys))
                if temp not in population:
                    population.append(temp)
                    i += 1
                else:
                    continue 
        return population


class GeneticAlgorithm:
    """Main genetic algorithm implementation for solving TSP with enhanced operators."""
    
    @staticmethod
    def get_fitness(genes, coordinates_2_cities):
        """
        Calculate fitness of a chromosome (inverse of total tour distance).
        Higher fitness indicates shorter tour distance.
        
        Args:
            genes (list): Chromosome representing tour sequence
            coordinates_2_cities (dict): Mapping from city index to coordinates
            
        Returns:
            float: Fitness value (1 / total_distance)
        """
        # Distance from last city back to first city (complete the tour)
        total_distance = Utils.get_euclidean_distance(
            coordinates_2_cities[genes[0]], 
            coordinates_2_cities[genes[-1]]
        )
        
        # Distance between consecutive cities in tour
        for i in range(len(genes) - 1):
            p1 = coordinates_2_cities[genes[i]]
            p2 = coordinates_2_cities[genes[i + 1]]
            total_distance += Utils.get_euclidean_distance(p1, p2)
        
        return 1 / total_distance

    @staticmethod
    def rank_population(population, coordinates_2_cities):
        """
        Rank population by fitness and calculate total fitness sum.
        
        Args:
            population (list): List of chromosomes
            coordinates_2_cities (dict): Mapping from city index to coordinates
            
        Returns:
            tuple: (ranked population as (fitness, chromosome) pairs, sum of all fitness values)
        """
        ranked_population = []
        sum_population_fitness = 0
        
        for chromosome in population:
            chromosome_fitness = GeneticAlgorithm.get_fitness(chromosome, coordinates_2_cities)
            ranked_population.append((chromosome_fitness, chromosome))
            sum_population_fitness += chromosome_fitness
            
        ranked_population.sort(key=lambda x: x[0], reverse=True)
        return ranked_population, sum_population_fitness

    @staticmethod
    def create_mating_pool(ranked_population, sum_population_fitness, elitism_rate):
        """
        Create mating pool using elitism and roulette wheel selection.
        Combines the best individuals (elitism) with probabilistic selection (roulette wheel).
        
        Args:
            ranked_population (list): Population ranked by fitness
            sum_population_fitness (float): Total fitness of population
            elitism_rate (float): Fraction of best individuals to preserve
            
        Returns:
            list: Mating pool of selected chromosomes
        """
        elite_size = int(len(ranked_population) * elitism_rate)
        mating_pool = [ranked_population[i][1] for i in range(elite_size)]

        # Roulette wheel selection for remaining spots
        for i in range(len(ranked_population) - elite_size):
            threshold = random.uniform(0, sum_population_fitness)
            partial_sum = 0
            for j in range(len(ranked_population)):
                partial_sum += ranked_population[j][0]
                if partial_sum >= threshold:
                    mating_pool.append(ranked_population[j][1])
                    break
        return mating_pool

    @staticmethod
    def two_point_crossover_over_population(mating_pool, elitism_rate):
        """
        Apply sophisticated two-point crossover strategy across the population.
        Uses a multi-tier approach: elite preservation, elite breeding, and general population breeding.
        
        Args:
            mating_pool (list): Pool of selected parent chromosomes
            elitism_rate (float): Fraction of elite individuals to preserve unchanged
            
        Returns:
            list: New population after crossover operations
        """
        elite_size = int(len(mating_pool) * elitism_rate)
        crossovered_population = []
        chromosome_size = len(mating_pool[0])

        # Tier 1: Preserve elite chromosomes unchanged
        for i in range(elite_size):
            crossovered_population.append(mating_pool[i])

        # Tier 2: Breed elite individuals with each other (high-quality offspring)
        for i in range(elite_size):
            random_idxs = random.sample(range(0, chromosome_size), 2)
            start_idx = min(random_idxs)
            end_idx = max(random_idxs)
            random_parent_ids = random.sample(range(0, elite_size), 2)
            
            # Create two children from elite parents
            child = CrossoverMethods.two_point_crossover(
                mating_pool[random_parent_ids[0]], 
                mating_pool[random_parent_ids[1]], 
                start_idx, end_idx
            )
            crossovered_population.append(child)
            
            child = CrossoverMethods.two_point_crossover(
                mating_pool[random_parent_ids[1]], 
                mating_pool[random_parent_ids[0]], 
                start_idx, end_idx
            )
            crossovered_population.append(child)

        # Tier 3: General population breeding with mutation
        temp_pool = random.sample(mating_pool, len(mating_pool))
        _iter = int((len(mating_pool) - (elite_size * 3)) / 2)
        
        for i in range(_iter):
            random_idxs = random.sample(range(0, chromosome_size), 2)
            start_idx = min(random_idxs)
            end_idx = max(random_idxs)
            random_parent_ids = random.sample(range(0, len(mating_pool)), 2)
            
            # Create children and immediately mutate them
            child = CrossoverMethods.two_point_crossover(
                temp_pool[random_parent_ids[0]], 
                temp_pool[random_parent_ids[1]], 
                start_idx, end_idx
            )
            crossovered_population.append(GeneticAlgorithm.mutate(child))
            
            child = CrossoverMethods.two_point_crossover(
                temp_pool[random_parent_ids[1]], 
                temp_pool[random_parent_ids[0]], 
                start_idx, end_idx
            )
            crossovered_population.append(GeneticAlgorithm.mutate(child))
            
        return crossovered_population

    @staticmethod
    def cycle_crossover_over_population(mating_pool, elitism_rate):
        """
        Apply cycle crossover across the entire mating pool with immediate mutation.
        
        Args:
            mating_pool (list): Pool of selected parent chromosomes
            elitism_rate (float): Fraction of elite individuals to preserve unchanged
            
        Returns:
            list: New population after cycle crossover
        """
        elite_size = int(len(mating_pool) * elitism_rate)
        crossovered_population = []

        # Preserve elite chromosomes
        for i in range(elite_size):
            crossovered_population.append(mating_pool[i])

        temp_pool = random.sample(mating_pool, len(mating_pool))
        _iter = int((len(mating_pool) - elite_size) / 2)

        # Generate offspring through cycle crossover with immediate mutation
        for i in range(_iter):
            random_parent_ids = random.sample(range(0, len(mating_pool)), 2)
            child1, child2 = CrossoverMethods.cycle_crossover(
                temp_pool[random_parent_ids[0]], 
                temp_pool[random_parent_ids[1]]
            )
            crossovered_population.append(GeneticAlgorithm.mutate(child1))
            crossovered_population.append(GeneticAlgorithm.mutate(child2))

        return crossovered_population

    @staticmethod
    def mutate(chromosome):
        """
        Mutate a chromosome using 2-opt-style inversion mutation.
        This is a TSP-specific mutation that reverses a segment of the tour,
        which can improve tour quality by eliminating edge crossings.
        
        Args:
            chromosome (list): Chromosome to mutate
            
        Returns:
            list: Mutated chromosome
        """
        chromosome_size = len(chromosome)
        random_idxs = random.sample(range(0, chromosome_size), 2)
        start_idx = min(random_idxs)
        end_idx = max(random_idxs)
        
        # Reverse the segment between start_idx and end_idx (2-opt style)
        chromosome[start_idx:end_idx] = chromosome[start_idx:end_idx][::-1]
        return chromosome

    @staticmethod
    def mutate_population(population):
        """
        Apply mutation to entire population using 2-opt inversion.
        
        Args:
            population (list): Population to mutate
            
        Returns:
            list: Population after mutation
        """
        mutated_population = []
        for chromosome in population:
            mutated_chromosome = GeneticAlgorithm.mutate(chromosome)
            mutated_population.append(mutated_chromosome)
        return mutated_population

    @staticmethod
    def get_next_generation(population, population_size, coordinates_2_cities, elitism_rate):
        """
        Generate the next generation through selection, crossover, and mutation.
        Uses a dual-population strategy combining crossover and mutation populations.
        
        Args:
            population (list): Current population
            population_size (int): Size of population to maintain
            coordinates_2_cities (dict): Mapping from city index to coordinates
            elitism_rate (float): Elitism rate for selection
            
        Returns:
            list: Next generation population
        """
        # Rank current population by fitness
        ranked_population, sum_population_fitness = GeneticAlgorithm.rank_population(
            population, coordinates_2_cities
        )
        
        # Create mating pool using elitism and roulette wheel selection
        mating_pool = GeneticAlgorithm.create_mating_pool(
            ranked_population[:population_size], sum_population_fitness, elitism_rate
        )
        
        # Apply crossover to create offspring population
        crossovered_population = GeneticAlgorithm.two_point_crossover_over_population(
            mating_pool, elitism_rate
        )
        
        # Create additional diversity through mutation
        temp_population = copy.deepcopy(crossovered_population)
        mutated_population = GeneticAlgorithm.mutate_population(temp_population)
        
        # Combine both populations for increased diversity
        population = mutated_population + crossovered_population
        return population


def solve_tsp(coordinates_2_cities, coordinates, population_size, elitism_rate, num_generations):
    """
    Main function to solve the Traveling Salesman Problem using genetic algorithm.
    Adaptively sets k-nearest neighbors parameter based on problem size.
    
    Args:
        coordinates_2_cities (dict): Mapping from city index to coordinates
        coordinates (list): List of city coordinates
        population_size (int): Size of genetic algorithm population
        elitism_rate (float): Fraction of best individuals to preserve each generation
        num_generations (int): Number of generations to evolve
    """
    # Adaptive k parameter based on problem size
    if len(coordinates_2_cities.keys()) <= 20:
        k = len(coordinates_2_cities.keys()) - 1  # Use all neighbors for small problems
    elif len(coordinates_2_cities.keys()) <= 50:
        k = 20  # Medium-sized problems
    else:
        k = 40  # Large problems
    
    # Initialize population using KNN heuristic
    population = PopulationInitializationMethods.knn_population_initialization(
        k, coordinates, coordinates_2_cities, population_size
    )
    
    if len(coordinates_2_cities.keys()) > 5:
        # Show initial best solution
        result, _ = GeneticAlgorithm.rank_population(population, coordinates_2_cities)
        print(f"Start shortest distance : {str(1 / result[0][0])}")
        print(f"Start shortest path : {result[0][1]}")
        
        # Evolve population for specified number of generations
        for i in range(num_generations):
            population = GeneticAlgorithm.get_next_generation(
                population, population_size, coordinates_2_cities, elitism_rate
            )
        
        # Show final best solution
        result, _ = GeneticAlgorithm.rank_population(population, coordinates_2_cities)
        print(f"Final shortest distance : {str(1 / result[0][0])}")
        print(f"Final shortest path : {result[0][1]}")
        Utils.write_output_file(result[0][1], coordinates_2_cities)
    else:
        # For very small problems, just find the best among all permutations
        result, _ = GeneticAlgorithm.rank_population(population, coordinates_2_cities)
        print(f"Shortest distance : {str(1 / result[0][0])}")
        print(f"Shortest path : {result[0][1]}")
        Utils.write_output_file(result[0][1], coordinates_2_cities)


def run(filepath):
    """
    Main execution function with optimized parameters.
    Reads input, configures algorithm parameters, and solves TSP with timing.
    """
    coordinates_2_cities, coordinates = Utils.map_coordinates_to_city_nums(filepath)
    population_size = 100
    elitism_rate = 0.2
    num_generations = 500
    solve_tsp(coordinates_2_cities, coordinates, population_size, elitism_rate, num_generations)


if __name__ == "__main__":
    start_time = time.time()
    run(filepath="./inputs/input3.txt")
    end_time = time.time()
    print(f"Execution Time : {end_time - start_time}")