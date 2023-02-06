"""
This code performs a feature selection using the TLBO algorithm to minimize a given fitness function. 
The user can define the problem parameters,
such as the number of iterations, population size, and the search space boundaries. 
The code returns the final population and their corresponding fitness values.
"""
import random
import numpy as np

# Define a fitness function
def fitness_function(x):
    return x**2 + x + 1

# TLBO function
def TLBO(fitness_function, dimension, n_iterations, population_size, a, b, p, q):
    # Initialize population
    population = np.random.uniform(a, b, (population_size, dimension))
    fitness = np.zeros(population_size)
    for i in range(population_size):
        fitness[i] = fitness_function(population[i])
    
    # Start TLBO loop
    for i in range(n_iterations):
        for j in range(population_size):
            # Select three random individuals
            indices = [k for k in range(population_size) if k != j]
            random_indices = random.sample(indices, 3)
            x1, x2, x3 = population[random_indices[0]], population[random_indices[1]], population[random_indices[2]]
            
            # Generate a new candidate
            mutant_vector = x1 + p * (x2 - x3)
            cross_over_point = np.random.randint(0, dimension)
            trial_vector = np.copy(population[j])
            trial_vector[:cross_over_point] = mutant_vector[:cross_over_point]
            
            # Calculate its fitness
            trial_fitness = fitness_function(trial_vector)
            
            # Update population
            if trial_fitness < fitness[j]:
                population[j] = trial_vector
                fitness[j] = trial_fitness
            else:
                if random.random() < q:
                    population[j] = trial_vector
    return population, fitness

# Define problem parameters
dimension = 2
n_iterations = 1000
population_size = 20
a, b = -10, 10
p, q = 0.5, 0.5

# Run TLBO
population, fitness = TLBO(fitness_function, dimension, n_iterations, population_size, a, b, p, q)
print("Final population:", population)
print("Final fitness:", fitness)