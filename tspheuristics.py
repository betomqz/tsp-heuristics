import numpy as np
import random
from mip import Model, xsum, minimize, BINARY
from itertools import product

class TabuSearch:
    '''
    Búsqueda tabú para resolver el TSP
    '''

    def __init__(self, instance, tabusize=10, maxiters=1000):
        self.instance = instance
        self.res = None
        self.cost = None
        self.tabusize = tabusize
        self.maxiters = maxiters

    def solve(self, verbose=True):
        '''
        Función para resolver un problema del TSP usando una
        búsqueda Tabú.
        '''

        # Generar solución inicial
        current_solution = self.__init_sol()
        
        # La mejor solución por ahora es la única que tenemos
        self.res = current_solution[:]
        self.cost = self.__calc_cost(self.res)
        
        # Arreglo para almacenar la lista tabú
        tabu_list = []
        iteration = 0

        if verbose:
            print(f"Solución inicial encontrada con costo: {self.cost} en iter {iteration}")
        
        while iteration < self.maxiters:
            
            # Generar vecindad alrededor de la solución actual
            neighborhood = self.__generate_neighborhood(current_solution)
            
            # El mejor vecino de la vecindad empieza vacío
            best_neighbor = None
            best_neighbor_distance = np.inf
            
            # Buscar al mejor vecino
            for neighbor in neighborhood:
                neighbor_distance = self.__calc_cost(neighbor)
                if neighbor not in tabu_list and neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
            
            # Si el mejor vecino es mejor que la mejor solución, actualizar la
            # mejor solución. De cualquier manera, guardarlo en la lista tabú.
            if best_neighbor:
                current_solution = best_neighbor[:]
                if best_neighbor_distance < self.cost:
                    self.res = best_neighbor[:]
                    self.cost = best_neighbor_distance
                    if verbose:
                        print(f"Solución inicial encontrada con costo: {self.cost} en iter {iteration}")
                tabu_list.append(best_neighbor)
                if len(tabu_list) > self.tabusize:
                    tabu_list.pop(0)
            
            iteration += 1
        
        self.res = self.res + [self.res[0]] # Regresar al inicio        
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")

    def __init_sol(self):
        '''
        Función para generar una primera solución. Utiliza la 
        heurística del vecino más cercano.
        '''
        nn = NearestNeigbour(self.instance)
        nn.solve(start=0, verbose=False)
        return nn.res[:-1]
     
    def __generate_neighborhood(self, solution):
        '''
        Función para generar una vecindad a partir de 
        una solución. Similar al de mutación porque intercambia
        dos ciudades de una solución dada.
        '''
        neighborhood = []
        num_cities = len(solution)
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                # Crear una copia de la solución
                neighbor = solution[:]

                # Intercambiar ciudades
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
                # Agregar a la vecindad
                neighborhood.append(neighbor)
        
        return neighborhood
    
    def __calc_cost(self, individual):
        '''
        Función para calcular el costo de una solución
        '''
        cost = 0
        for i in range(len(individual)-1):
            cost += self.instance[individual[i], individual[i+1]]

        cost += self.instance[individual[-1], individual[0]]
        return cost


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
    def __init__(self, instance, pop_size=50, mutation_rate=0.05, num_gens=2000, use_nn_seed=True):
        self.instance = instance
        self.res = None
        self.cost = None
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.num_gens = num_gens
        self.__population = None
        self.use_nn_seed = use_nn_seed
        self.__nn_seed = None
    
    def solve(self, verbose=True):
        '''
        Resuelve el problema del TSP con un algoritmo genético. Utiliza los 
        parámetros que se definieron al inicializar la clase.
        '''
        # inicializar población        
        self.__init_pop()

        for gen in range(self.num_gens):
            for i in range(self.pop_size-1):
                # seleccionar a los padres
                parent1 = self.__population[i]
                parent2 = self.__population[i+1]
                # Crear al hijo
                child = self.__crossover(parent1, parent2)
                
                # Mutar al hijo si es necesario
                if random.random() < self.mutation_rate:
                    child = self.__mutate(child)

                # Insertar al hijo en la población 
                self.__population.append(child)

            # ordenar a la población y seleccionar a los mejores
            self.__population.sort(key=self.__calc_cost)
            self.__population = self.__population[:self.pop_size]

            if gen % 10 == 0 and verbose:
                best_individual = self.__population[0]
                print(f"Generación {gen}: {self.__calc_cost(best_individual)}")

        # escoger al mejor individuo
        best_individual = self.__population[0]
        self.res = best_individual + [best_individual[0]] # Regresar al inicio
        self.cost = self.__calc_cost(best_individual)
        if verbose:
            print(f"Solución encontrada con costo: {self.cost}")

    def __init_pop(self):
        '''
        Función para inicializar la población. Inicializa aleatoriamente 
        o con la solución del vecino más cercano, como se indique, y
        luego los ordena según el costo de cada solución.
        '''
        if self.use_nn_seed:
            nn = NearestNeigbour(self.instance)
            nn.solve(start=0, verbose=False)
            self.__nn_seed = nn.res[:-1]
            self.__population = [self.__nn_seed]
            for i in range(self.pop_size-1):
                individual = self.__mutate(self.__nn_seed)
                self.__population.append(individual)
        else:
            self.__population = []
            num_cities = len(self.instance)
            for _ in range(self.pop_size):
                individual = list(range(num_cities))
                random.shuffle(individual)
                self.__population.append(individual)
        
        self.__population.sort(key=self.__calc_cost)

    def __mutate(self, individual):
        '''
        Función para mutar a un individuo. Escoge dos ciudades aleatoriamente
        y las intercambia.
        '''
        ind_copy = individual.copy()
        idx1, idx2 = random.sample(range(len(ind_copy)), 2)
        ind_copy[idx1], ind_copy[idx2] = ind_copy[idx2], ind_copy[idx1]
        return ind_copy
    
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

    def __calc_cost(self, individual):
        cost = 0
        for i in range(len(individual)-1):
            cost += self.instance[individual[i], individual[i+1]]

        cost += self.instance[individual[-1], individual[0]]
        return cost
