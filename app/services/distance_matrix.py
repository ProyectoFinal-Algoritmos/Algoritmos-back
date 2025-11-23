import networkx as nx
from typing import List
from app.models.distance_matrix_result import DistanceMatrixResult
from typing import Optional





_last_distance_matrix: Optional[DistanceMatrixResult] = None

def get_distance_matrix() -> DistanceMatrixResult:
    if _last_distance_matrix is None:
        raise ValueError("Distance matrix has not been built yet.")
    return _last_distance_matrix

def set_distance_matrix(matrix: DistanceMatrixResult):
    global _last_distance_matrix
    _last_distance_matrix = matrix


#construye la matriz de distancia para usar en los algoritmos
#recibe un grafo y la lista de nodos que se quieren visitar, retorna un objeto con matriz de distancias y caminos
    #lo hace con shortest path
def build_distance_matrix_with_paths(G: nx.Graph, node_ids: List[int]
) -> DistanceMatrixResult:
    
    n = len(node_ids)
    #se inicializa en 0
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    #se inicializa con listas vacías
    paths = [[[] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            #cuando un nodo se evalúa a si mismo
            if i == j:
                distances[i][j] = 0.0
                paths[i][j] = [node_ids[i]]
            else:
                try:
                    #funciones shortest path -> usan Dijkstra dado que se le pasa weight. Si no se le pasa usa BFS
                    path = nx.shortest_path(G, source=node_ids[i], target=node_ids[j], weight="weight") #usa atributo llamado weight en la arista
                    dist = nx.shortest_path_length(G, source=node_ids[i], target=node_ids[j], weight="weight")
                    distances[i][j] = dist
                    paths[i][j] = path
                    #si no hay un camino hasta ese nodo 
                except nx.NetworkXNoPath:
                    distances[i][j] = float('inf')
                    paths[i][j] = []

    return DistanceMatrixResult(distances=distances, paths=paths)
