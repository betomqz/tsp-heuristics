import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

# Función para generar la matriz de distancias de un arreglo de coordenadas
def dist_mat(coords):
    return squareform(pdist(coords))

# Función para calcular el costo (o distancia) de un camino dado.
def calc_cost(path, distances):
    cost = 0
    for i in range(len(path)-1):
        cost += distances[path[i], path[i+1]]

    return cost


# Función para graficar las soluciones al problema
def plot_sol(puntos, path, savefig=""):
    x, y = puntos[path].T
    names = np.array(range(len(puntos)))
    names2 = names[path]

    fig, ax = plt.subplots(figsize=(5,5))    
    ax.plot(x,y,'-o')
    ax.axis('equal')
    
    for i, txt in enumerate(names2):
        ax.annotate(txt, (x[i], y[i]), xytext=(2, 5), textcoords='offset points')

    if savefig != "":
        plt.savefig("imgs/"+savefig)

# Función para leer una instancia del folder
def read_instance(instance):
    file_path = os.path.join('instances', instance)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        cities = []
        for line in lines[5:]:
            if line.strip() == "EOF":
                break
            city = line.split()
            cities.append([float(city[1]), float(city[2])])
    return np.array(cities)
