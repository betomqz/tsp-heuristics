import numpy as np
import random

def calculate_distance(cities, order):
    distance = 0
    num_cities = len(order)
    for i in range(num_cities):
        from_city = order[i]
        to_city = order[(i + 1) % num_cities]
        distance += np.linalg.norm(cities[from_city] - cities[to_city])
    return distance

def generate_initial_solution(num_cities):
    return list(range(num_cities))

def generate_neighborhood(solution):
    neighborhood = []
    num_cities = len(solution)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            neighbor = solution[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighborhood.append(neighbor)
    return neighborhood

def tabu_search(cities, max_iterations, tabu_size):
    num_cities = len(cities)
    current_solution = generate_initial_solution(num_cities)
    best_solution = current_solution[:]
    tabu_list = []
    iteration = 0
    
    while iteration < max_iterations:
        neighborhood = generate_neighborhood(current_solution)
        best_neighbor = None
        best_neighbor_distance = float('inf')
        
        for neighbor in neighborhood:
            neighbor_distance = calculate_distance(cities, neighbor)
            if neighbor not in tabu_list and neighbor_distance < best_neighbor_distance:
                best_neighbor = neighbor
                best_neighbor_distance = neighbor_distance
        
        if best_neighbor:
            current_solution = best_neighbor[:]
            if best_neighbor_distance < calculate_distance(cities, best_solution):
                best_solution = best_neighbor[:]
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        
        iteration += 1
    
    return best_solution, calculate_distance(cities, best_solution)

# Example usage:
# Define the cities as coordinates
cities = np.array([[0, 0], [1, 2], [3, 1], [5, 2]])

# Set parameters
max_iterations = 1000
tabu_size = 10

# Run Tabu Search
best_solution, best_distance = tabu_search(cities, max_iterations, tabu_size)
print("Best solution:", best_solution)
print("Best distance:", best_distance)
