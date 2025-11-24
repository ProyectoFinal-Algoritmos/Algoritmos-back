from typing import List
from typing import List

#pasa de id por numero de arreglo al id real del nodo en el grafo
def map_path_indices_to_ids(path: List[int], node_ids: List[int]) -> List[int]:
    #para cada casilla se le asigna el resultado de node_ids en la casilla del nodo a buscar
    return [node_ids[i] for i in path]




#reconstruye ruta segun un camino de indices
def reconstruct_full_path(index_path: List[int], path_matrix: List[List[List[int]]]) -> List[int]:
    
    full_path = []

    for i in range(len(index_path) - 1):
        a, b = index_path[i], index_path[i + 1]
        #se obtiene el camino desde el nodo a hasta el b segun la matriz
        segment = path_matrix[a][b]

        if i > 0 and full_path[-1] == segment[0]:
            #para no duplicar el nodo que conecta con el siguiente
            segment = segment[1:]

        full_path.extend(segment)

    return full_path
