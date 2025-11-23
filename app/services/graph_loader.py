import networkx as nx #para grafos
from geopy.distance import geodesic #para distancias

import osmnx as ox #para cargar archivo OSM en el grafo

from typing import BinaryIO

#al inicio es nulo pero varia para el grafo
_loaded_graph: nx.Graph | None = None


def get_graph() -> nx.Graph:
    if _loaded_graph is None:
        raise ValueError("No graph has been loaded yet.")
    return _loaded_graph

#recibe y carga el grafo desde un archivo XML
import networkx as nx
import osmnx as ox
from geopy.distance import geodesic
from typing import BinaryIO
import tempfile

_loaded_graph: nx.Graph | None = None

def get_graph() -> nx.Graph:
    if _loaded_graph is None:
        raise ValueError("No graph has been loaded yet.")
    return _loaded_graph

#carga el grafo desde un archivo XML recibido 
def load_graph_from_file(file: BinaryIO) -> dict:
    global _loaded_graph

    # Guardar el archivo en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".osm") as temp:
        temp.write(file.read())
        temp_path = temp.name

    # Carga el grafo desde el archivo temporal
    G = ox.graph_from_xml(temp_path, simplify=True)

    if G.is_directed():
        G = G.to_undirected()

    for node_id, data in G.nodes(data=True):
        data["latitude"] = data.get("y")
        data["longitude"] = data.get("x")

    for u, v, data in G.edges(data=True):
        node_1 = G.nodes[u]
        node_2 = G.nodes[v]
        coord_1 = (node_1["latitude"], node_1["longitude"])
        coord_2 = (node_2["latitude"], node_2["longitude"])
        data["weight"] = geodesic(coord_1, coord_2).meters

    _loaded_graph = G

    return {
        "status": "success",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges()
    }






def get_graph_data() -> dict:
    if _loaded_graph is None:
        raise ValueError("Graph not loaded yet.")

    nodes = [
        {"id": nid, "lat": data["latitude"], "lon": data["longitude"]}
        for nid, data in _loaded_graph.nodes(data=True)
    ]

    edges = [
        {"from": u, "to": v}
        for u, v in _loaded_graph.edges()
    ]

    return {"nodes": nodes, "edges": edges}

