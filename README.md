# Heurísticas para resolver el TSP

Este repositorio contiene dos heurísticas para resolver el TSP: la heurística del vecino más cercano y un algoritmo genético. Ambas heurísticas se prueban con las instancias contenidas en [`instances/`](/instances/).

## Estructura del código
- La programación de las heurísticas se encuentra en el módulo [`tspheuristics.py`](/tspheuristics.py). Este módulo contiene dos clases: `NearestNeigbour` y `GenAlgo`.
- Las pruebas se hacen en el notebook [`arena.ipynb`](/arena.ipynb). La idea es crear varias instancias de las clases y probarlas con las distintas instancias del TSP.
- El módulo [`utils.py`](/utils.py) contiene algunas funciones útiles, como para leer una instancia del TSP o graficar una solución.
- El archivo [`requirements.txt`](/requirements.txt) contiene los paquetes y las versiones utilizadas para desarrollar este código.