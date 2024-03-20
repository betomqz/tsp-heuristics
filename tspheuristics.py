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
        # número de nodos y lista de vértices
        n = len(self.instance)
        V = set(range(len(self.instance)))

        model = Model()

        # variables binarias para indicar si el camino (i,j) se usa en la ruta
        x = [[model.add_var(var_type=BINARY) for j in V] for i in V]

        # variables para prevenir subtours: cada ciudad tendrá una
        # secuencia diferente de ids en la ruta, excepto la primera        
        y = [model.add_var() for i in V]

        # definir función objetivo
        model.objective = minimize(xsum(self.instance[i][j]*x[i][j] for i in V for j in V))

        # que solamente un camino salga de una ciudad
        for i in V:
            model += xsum(x[i][j] for j in V - {i}) == 1

        # que solamente llegue un camino a una ciudad
        for i in V:
            model += xsum(x[j][i] for j in V - {i}) == 1

        # eliminar subtours
        for (i, j) in product(V - {0}, V - {0}):
            if i != j:
                model += y[i] - (n+1)*x[i][j] >= y[j]-n

        # resolver el problema
        model.optimize()

        # revisar si se encuentra una solución
        if model.num_solutions:
            # almacenar costo de la solución
            self.cost = model.objective_value

            # extraer el camino de solución
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
    '''
    Heurística del vecino más cercano para resolver el TSP.
    '''
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
        # Empezar en la ciudad que se indica
        self.res = [start]
        visited[start] = True        
        current = start
        self.cost = 0

        # mientras no haya recorrido todas las ciudades
        while len(self.res) < n:
            nearest_city = None
            min_distance = np.inf # La distancia mínima empieza en infinito
            for next in range(n):
                # Si encuentra una ciudad que está más cerca
                if not visited[next] and self.instance[current][next] < min_distance:
                    nearest_city = next
                    min_distance = self.instance[current][next]
            self.res.append(nearest_city) # almacenar ciudad
            visited[nearest_city] = True
            self.cost += min_distance
            current = nearest_city

        self.res.append(start) # regresar al principio
        self.cost += self.instance[current][start] # agregar costo de regresar
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")
    
class GenAlgo:
    '''
    Metaheurística de un algoritmo genético para resolver el TSP.
    '''
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
        Resuelve el problema del TSP con un algoritmo genético. Utiliza los parámetros que se definieron
        al inicializar la clase.
        '''
        self.__init_pop() # inicializar población
        for gen in range(self.num_gens): 
            new_population = [] # la nueva población empieza vacía
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.__select_parents() # seleccionar a los padres
                # Crear a los hijos
                child1 = self.__crossover(parent1, parent2)
                child2 = self.__crossover(parent2, parent1)
                # Mutar a los hijos si es necesario
                if random.random() < self.mutation_rate:
                    child1 = self.__mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.__mutate(child2)

                # Insertar a los hijos en la población 
                new_population.extend([child1, child2])
            self.__population = new_population
            if gen % 10 == 0 and verbose:
                best_individual = min(self.__population, key=lambda x: utils.calc_cost_gen(x, self.instance))
                print(f"Generación {gen}: {utils.calc_cost_gen(best_individual, self.instance)}")

        # escoger al mejor individuo
        best_individual = min(self.__population, key=lambda x: utils.calc_cost_gen(x, self.instance))
        self.res = best_individual + [best_individual[0]] # Regresar al inicio
        self.cost = utils.calc_cost_gen(best_individual, self.instance)
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")

    def __init_pop(self):
        '''
        Función para inicializar la población. Inicializa aleatoriamente.
        '''
        self.__population = []
        num_cities = len(self.instance)
        for _ in range(self.pop_size):
            individual = list(range(num_cities))
            random.shuffle(individual)
            self.__population.append(individual)

    def __mutate(self, individual):
        '''
        Función para mutar a un individuo. Escoge dos ciudades aleatoriamente
        y las intercambia.
        '''
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def __crossover(self, parent1, parent2):
        '''
        Función para generar un individuo dados dos padres. Hace un OX
        crossover: toma un pedazo del primer padre y luego llena los 
        huecos como haga falta.
        '''
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
        '''
        Función para seleccionar a los padres. 
        '''
        parents = []
        for _ in range(2):
            tournament = random.sample(self.__population, tournament_size)
            parents.append(min(tournament, key=lambda x: utils.calc_cost_gen(x, self.instance)))
        return parents
