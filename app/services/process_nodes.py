import networkx as nx

from shapely.geometry import Point, LineString
from geopy.distance import geodesic


from typing import List, Tuple, BinaryIO

_selected_node_ids: List[int] = []

def get_selected_nodes() -> List[int]:
    return _selected_node_ids

#En esta funcion se lee un archivo con los puntos a visitar
def load_points_from_uploaded_file(file: BinaryIO, G: nx.Graph) -> List[int]:
    content = file.read().decode("utf-8")
    lines = content.strip().splitlines()
    
    points = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        lat = float(parts[0])
        lon = float(parts[1])
        points.append((lat, lon))

    node_ids = process_points_into_graph(G, points)

    global _selected_node_ids
    _selected_node_ids = node_ids

    return node_ids





#procesa cada nuevo nodo para validar si se debe insertar en el grafo
def process_points_into_graph(G: nx.Graph, points: List[Tuple[float, float]]
) -> List[int]:
    
    result_node_ids = []
    #recorre nodos nuevos
    for lat, lon in points:
        # Busca si el nodo ya existe según sus coordenadas
        existing = None
        #recorre los nodos que ya existen en el grafo
        for nid, data in G.nodes(data=True):
            if data.get("latitude") == lat and data.get("longitude") == lon:
                existing = nid
                break

        if existing is not None:
            #no se inserta en el grafo pero si se agrega a la lista de nodos a recorrer
            result_node_ids.append(existing)
        else:
            # si el nodo no existe actualmente en el grafo, lo debe insertar
            # se genera automáticamente un nuevo ID secuencial
            new_id = max(G.nodes) + 1 if G.number_of_nodes() > 0 else 0
            inserted_id = insert_node_into_graph(G, new_id, lat, lon)
            #agrega el nuevo nodo a la lista de nodos a recorrer
            result_node_ids.append(inserted_id)

    #retorno ids de todos los nodos que se tienen que recorrer
    return result_node_ids




#inserta nuevos nodos en el grafo, para eso busca la arista más cercana y la divide en 2
def insert_node_into_graph(G: nx.Graph, new_id: int, lat: float, lon: float) -> int:
  
    #shapely espera coordenadas como (x, y) = (lon, lat)
    new_point = Point(lon, lat)
    closest_edge = None
    min_distance = float("inf")
    best_line = None  # guardará la geometría de la arista más cercana

    # Buscar la arista más cercana (por distancia perpendicular)
    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        # LineString con (lon, lat)
        line = LineString([
            (u_data["longitude"], u_data["latitude"]),
            (v_data["longitude"], v_data["latitude"])
        ])
        distance = new_point.distance(line)

        if distance < min_distance:
            min_distance = distance
            closest_edge = (u, v)
            best_line = line

    if closest_edge is None or best_line is None:
        raise ValueError("No edge found to insert point")

    u, v = closest_edge
    coord_u = (G.nodes[u]["latitude"], G.nodes[u]["longitude"])
    coord_v = (G.nodes[v]["latitude"], G.nodes[v]["longitude"])

    # proyectar el punto sobre la línea más cercana
    projected = best_line.interpolate(best_line.project(new_point))
    new_coord = (projected.y, projected.x)  # convertir de (x, y) a (lat, lon)

    # se elimina la arista original (ya que se debe dividir en 2)
    G.remove_edge(u, v)

    # se inserta el nuevo nodo
    G.add_node(new_id, latitude=new_coord[0], longitude=new_coord[1])

    # Conectar los extremos de la anterior arista con el nuevo nodo 
    dist_u = geodesic(coord_u, new_coord).meters
    dist_v = geodesic(new_coord, coord_v).meters
    G.add_edge(u, new_id, weight=dist_u)
    G.add_edge(new_id, v, weight=dist_v)

    return new_id
