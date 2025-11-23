import time
from app.models.tsp_result import TSPResult
from typing import List
from itertools import permutations


def solve_tsp_brute_force(distance_matrix: List[List[float]], start_index: int = 0) -> TSPResult:
    n = len(distance_matrix)
    nodes = list(range(n))
    nodes.remove(start_index)

    best_path = []
    min_cost = float('inf')

    start_time = time.time()

    #para poder calcular todas las combinaciones en las que se puede visitar los nodos
    for perm in permutations(nodes):
        path = [start_index] + list(perm) + [start_index]
        cost = 0.0

        #suma el costo total del camino que se está evaluando
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            cost += distance_matrix[a][b]

        #si es la mejor en terminos de distancia, la cambia
        if cost < min_cost:
            min_cost = cost
            best_path = path

    end_time = time.time()
    #tiempo total que se demoró el algoritmo
    execution_time = end_time - start_time

    return TSPResult(path=best_path, total_cost=min_cost, execution_time=execution_time, algorithmName = "Fuerza bruta")
  


#ejecuta TSP usando programación dinámica, Mediante held-karp 
def solve_tsp_dynamic_programming(distance_matrix: List[List[float]], start_index: int = 0) -> TSPResult:
    n = len(distance_matrix)
    ALL_VISITED = (1 << n) - 1
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1 << start_index][start_index] = 0
    start_time = time.time()

    for mask in range(1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0 and last != start_index:
                continue
            if prev_mask == 0 and last == start_index:
                continue
            for k in range(n):
                if not (prev_mask & (1 << k)):
                    continue
                cost = dp[prev_mask][k] + distance_matrix[k][last]
                if cost < dp[mask][last]:
                    dp[mask][last] = cost
                    parent[mask][last] = k

    min_cost = float('inf')
    last_index = -1

    for i in range(n):
        if i == start_index:
            continue
        cost = dp[ALL_VISITED][i] + distance_matrix[i][start_index]

        if cost < min_cost:
            min_cost = cost
            last_index = i

    path = [start_index]
    mask = ALL_VISITED
    curr = last_index
    
    for _ in range(n - 1):
        path.append(curr)
        temp = parent[mask][curr]
        mask ^= (1 << curr)
        curr = temp

    path.append(start_index)
    path.reverse()
    end_time = time.time()
    execution_time = end_time - start_time
    return TSPResult(path=path, total_cost=min_cost, execution_time=execution_time, algorithmName="Programacion Dinamica (Held-Karp)")


#ejecuta TSP usando algoritmo Greedy
def solve_tsp_greedy(distance_matrix: List[List[float]], start_index: int = 0) -> TSPResult:
    n = len(distance_matrix)
    visited = [False] * n
    path = [start_index]
    visited[start_index] = True
    total_cost = 0.0
    
    start_time = time.time()
    
    current_node = start_index
    
    # Visitar n-1 nodos restantes seleccionando siempre el más cercano no visitado
    for _ in range(n - 1):
        min_distance = float('inf')
        next_node = -1
        
        # Buscar el nodo más cercano que no ha sido visitado
        for j in range(n):
            if not visited[j] and distance_matrix[current_node][j] < min_distance:
                min_distance = distance_matrix[current_node][j]
                next_node = j
        
        # Moverse al siguiente nodo
        path.append(next_node)
        visited[next_node] = True
        total_cost += min_distance
        current_node = next_node
    
    # Regresar al nodo inicial para completar el ciclo
    path.append(start_index)
    total_cost += distance_matrix[current_node][start_index]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return TSPResult(path=path, total_cost=total_cost, execution_time=execution_time, algorithmName="Greedy (Vecino mas cercano)")