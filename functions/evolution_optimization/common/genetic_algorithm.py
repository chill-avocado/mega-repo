# Common genetic_algorithm component for evolution_optimization
class GeneticAlgorithm:
    """Simple genetic algorithm implementation."""
    
    def __init__(self, population_size=100, mutation_rate=0.01, crossover_rate=0.7):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size (int): Size of the population.
            mutation_rate (float): Probability of mutation (0-1).
            crossover_rate (float): Probability of crossover (0-1).
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
    
    def initialize(self, create_individual):
        """
        Initialize the population.
        
        Args:
            create_individual (callable): Function that creates a random individual.
        """
        self.population = [create_individual() for _ in range(self.population_size)]
        self.generation = 0
    
    def evolve(self, fitness_func, selection_func, crossover_func, mutation_func, generations=1):
        """
        Evolve the population for a number of generations.
        
        Args:
            fitness_func (callable): Function that evaluates an individual's fitness.
            selection_func (callable): Function that selects parents based on fitness.
            crossover_func (callable): Function that creates offspring from parents.
            mutation_func (callable): Function that mutates an individual.
            generations (int): Number of generations to evolve.
        
        Returns:
            tuple: Best individual and its fitness.
        """
        import random
        
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness_func(individual) for individual in self.population]
            
            # Find best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            best_individual = self.population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(best_individual)
            
            # Create the rest of the new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = selection_func(self.population, fitness_scores)
                parent2 = selection_func(self.population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = crossover_func(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = mutation_func(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = mutation_func(offspring2)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population = new_population
            self.generation += 1
        
        # Evaluate final fitness
        fitness_scores = [fitness_func(individual) for individual in self.population]
        best_idx = fitness_scores.index(max(fitness_scores))
        
        return self.population[best_idx], fitness_scores[best_idx]
