"""
Evolution and optimization interfaces for the integrated system.

This module defines the interfaces for evolution and optimization components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from .base import Component, Configurable, Observable


class EvolutionOptimizer(Component, Configurable, Observable):
    """Interface for evolution and optimization components in the integrated system."""

    @abstractmethod
    def evolve(self, population: List[Dict[str, Any]], generations: int, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evolve a population over multiple generations.

        Args:
            population: List of dictionaries representing individuals in the population.
            generations: Number of generations to evolve.
            options: Optional dictionary containing evolution options.

        Returns:
            Dictionary containing the evolved population and metadata.
        """
        pass

    @abstractmethod
    def evaluate(self, individual: Dict[str, Any]) -> float:
        """
        Evaluate the fitness of an individual.

        Args:
            individual: Dictionary representing an individual.

        Returns:
            Fitness score of the individual.
        """
        pass

    @abstractmethod
    def select(self, population: List[Dict[str, Any]], count: int, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Select individuals from a population.

        Args:
            population: List of dictionaries representing individuals in the population.
            count: Number of individuals to select.
            options: Optional dictionary containing selection options.

        Returns:
            List of selected individuals.
        """
        pass

    @abstractmethod
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform crossover between two parents to create a child.

        Args:
            parent1: Dictionary representing the first parent.
            parent2: Dictionary representing the second parent.
            options: Optional dictionary containing crossover options.

        Returns:
            Dictionary representing the child.
        """
        pass

    @abstractmethod
    def mutate(self, individual: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Mutate an individual.

        Args:
            individual: Dictionary representing the individual to mutate.
            options: Optional dictionary containing mutation options.

        Returns:
            Dictionary representing the mutated individual.
        """
        pass

    @abstractmethod
    def set_fitness_function(self, fitness_function: Callable[[Dict[str, Any]], float]) -> bool:
        """
        Set the fitness function for evaluating individuals.

        Args:
            fitness_function: Function that takes an individual and returns a fitness score.

        Returns:
            True if the fitness function was set successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_best_individual(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the best individual from a population.

        Args:
            population: List of dictionaries representing individuals in the population.

        Returns:
            Dictionary representing the best individual.
        """
        pass