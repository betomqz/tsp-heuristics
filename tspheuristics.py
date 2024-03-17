import numpy as np

class NearestNeigbour:
    def __init__(self, instance):
        self.instance = instance
        self.res = None
        self.cost = None

    def solve(self, start):
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
        return self.res, self.cost
    
class GenAlgo:
    def __init__(self, instance):
        self.instance = instance
        self.res = None
        self.cost = None
    
    def solve(self):
        pass
