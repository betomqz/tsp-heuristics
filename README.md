# Heurísticas para resolver el TSP

Este repositorio contiene dos heurísticas para resolver el TSP: la heurística del vecino más cercano y un algoritmo genético. Ambas heurísticas se prueban con las instancias contenidas en [`instances/`](/instances/).

## Estructura del código
- La programación de las heurísticas se encuentra en el módulo [`tspheuristics.py`](/tspheuristics.py). Este módulo contiene cuatro clases: `TabuSearch`, `LinProg`, `NearestNeigbour` y `GenAlgo`.
- El módulo [`utils.py`](/utils.py) contiene algunas funciones útiles, como para leer una instancia del TSP o graficar una solución.
- El archivo [`requirements.txt`](/requirements.txt) contiene los paquetes y las versiones utilizadas para desarrollar este código.
- Las pruebas se hacen en el notebook [`arena.ipynb`](/arena.ipynb). La idea es crear varias instancias de las clases y probarlas con las distintas instancias del TSP.

## Ejemplo de ejecución
Para probar el código, se puede hacer una ejecución como la siguiente:

```{python}
import numpy as np
import utils
import time
from tspheuristics import NearestNeigbour, GenAlgo, LinProg, TabuSearch

cities = utils.read_instance("Inst29.txt")
tsp_inst = utils.dist_mat(cities)

# Crear 3 objetos para las 3 formas de resolver el problema
tabu = TabuSearch(tsp_inst, tabusize=20, maxiters=2000)
gen = GenAlgo(tsp_inst, pop_size=len(cities)*2)
mymip = LinProg(tsp_inst)

# Resolver
tabu.solve()
gen.solve()
mymip.solve()

# Graficar las soluciones
utils.plot_sol(cities, tabu.res, savefig=f"tabu-sol.png")
utils.plot_sol(cities, gen.res, savefig=f"gen-sol.png")
utils.plot_sol(cities, mymip.res, savefig=f"mip-sol.png")
```