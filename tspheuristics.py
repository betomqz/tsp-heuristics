import numpy as np
import random
import utils
from mip import Model, xsum, minimize, BINARY
from itertools import product

class LinProg:
    '''
    Aunque no es una heurística, se incluye aquí el código para resolver el
    problema con una formulación de programación lineal. El código se adapta
    de https://docs.python-mip.com/en/latest/examples.html#the-traveling-salesman-problem
    '''

    def __init__(self, instance):
        self.instance = instance
        self.res = None
        self.cost = None
    
    def solve(self, verbose=True):
        # number of nodes and list of vertices
        n = len(self.instance)
        V = set(range(len(self.instance)))

        model = Model()

        # binary variables indicating if arc (i,j) is used on the route or not
        x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

        # continuous variable to prevent subtours: each city will have a
        # different sequential id in the planned route except the first one
        y = [model.add_var() for i in V]

        # objective function: minimize the distance
        model.objective = minimize(xsum(self.instance[i][j]*x[i][j] for i in V for j in V))

        # constraint : leave each city only once
        for i in V:
            model += xsum(x[i][j] for j in V - {i}) == 1

        # constraint : enter each city only once
        for i in V:
            model += xsum(x[j][i] for j in V - {i}) == 1

        # subtour elimination
        for (i, j) in product(V - {0}, V - {0}):
            if i != j:
                model += y[i] - (n+1)*x[i][j] >= y[j]-n

        # optimizing
        model.optimize()

        # checking if a solution was found
        if model.num_solutions:
            self.cost = model.objective_value

            nc = 0
            self.res = [0]
            while True:
                nc = [i for i in V if x[nc][i].x >= 0.99][0]
                self.res.append(nc)
                if nc == 0:
                    break
            if verbose:
                print(f"Solución encontrada con costo: {self.cost}")


class NearestNeigbour:
    def __init__(self, instance):
        self.instance = instance
        self.res = None
        self.cost = None

    def solve(self, start, verbose=True):
        '''
        Función con la heurística propiamente. Toma un punto de partida
        y resuelve la instancia del TSP utilizada al momento de crear.
        Regresa todo un camino y el costo de dicho camino.
        '''
        n = len(self.instance)
        visited = [False] * n
        self.res = [start]
        visited[start] = True

        current = start
        self.cost = 0

        while len(self.res) < n:
            nearest_city = None
            min_distance = np.inf # La distancia mínima empieza en infinito
            for next in range(n):
                if not visited[next] and self.instance[current][next] < min_distance:
                    nearest_city = next
                    min_distance = self.instance[current][next]
            self.res.append(nearest_city)
            visited[nearest_city] = True
            self.cost += min_distance
            current = nearest_city

        self.res.append(start)
        self.cost += self.instance[current][start]
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")
    
class GenAlgo:
    def __init__(self, instance, pop_size=50, mutation_rate=0.05, num_gens=2000, tournament_size=5):
        self.instance = instance
        self.res = None
        self.cost = None
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.num_gens = num_gens
        self.tournament_size = tournament_size
        self.__population = None
    
    def solve(self, verbose=True):
        '''
        Resuelve el problema del TSP con un algoritmo genético
        '''
        self.__init_pop()
        for gen in range(self.num_gens):
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.__select_parents()
                child1 = self.__crossover(parent1, parent2)
                child2 = self.__crossover(parent2, parent1)
                if random.random() < self.mutation_rate:
                    child1 = self.__mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.__mutate(child2)
                new_population.extend([child1, child2])
            self.__population = new_population
            if gen % 10 == 0 and verbose:
                best_individual = min(self.__population, key=lambda x: utils.calc_cost_gen(x, self.instance))
                print(f"Generación {gen}: {utils.calc_cost_gen(best_individual, self.instance)}")

        best_individual = min(self.__population, key=lambda x: utils.calc_cost_gen(x, self.instance))
        self.res = best_individual + [best_individual[0]] # Regresar al inicio
        self.cost = utils.calc_cost_gen(best_individual, self.instance)
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")

    def __init_pop(self):
        '''
        Función para inicializar la población.
        '''
        self.__population = []
        num_cities = len(self.instance)
        for _ in range(self.pop_size):
            individual = list(range(num_cities))
            random.shuffle(individual)
            self.__population.append(individual)

    def __mutate(self, individual):
        # Swap mutation
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def __crossover(self, parent1, parent2):
        # Order crossover (OX1)
        # No estoy seguro de que genere una solución válida tbh
        child = [-1] * len(parent1)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        for i in range(start, end + 1):
            child[i] = parent1[i]

        remaining = [item for item in parent2 if item not in child]
        ptr = 0
        for i in range(len(parent1)):
            if child[i] == -1:
                child[i] = remaining[ptr]
                ptr += 1
        return child
    
    def __select_parents(self, tournament_size=5):
        # De los individuos, escoge al mejor y lo regresa como padre. Hace esto dos veces
        parents = []
        for _ in range(2):
            tournament = random.sample(self.__population, tournament_size)
            parents.append(min(tournament, key=lambda x: utils.calc_cost_gen(x, self.instance)))
        return parents
