import math
import xml.etree.ElementTree as ET
from typing import Dict, Any

import networkx as nx

# Grafo en memoria (compartido entre requests)
_GRAPH: nx.Graph | None = None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia aproximada en metros entre dos coordenadas (lat, lon).
    """
    R = 6371000  # radio Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _parse_osm_to_graph(file_path: str) -> nx.Graph:
    """
    Lee un archivo .osm (XML) y construye un grafo no dirigido:
    - nodos: id OSM con atributos lat, lon, latitude, longitude
    - aristas: entre nodos consecutivos de cada <way>, con peso = distancia
    """
    print(f"Parsing OSM file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    G = nx.Graph()

    # 1) Cargar todos los nodos <node id="..." lat="..." lon="...">
    for node in root.findall("node"):
        node_id = int(node.attrib["id"])
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])

        # Guardamos con ambos nombres para ser compatibles
        G.add_node(
            node_id,
            lat=lat,
            lon=lon,
            latitude=lat,
            longitude=lon,
        )

    # 2) Para cada <way>, crear aristas entre nodos consecutivos
    for way in root.findall("way"):
        nd_refs = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
        for u, v in zip(nd_refs, nd_refs[1:]):
            if G.has_node(u) and G.has_node(v):
                lat1 = G.nodes[u]["lat"]
                lon1 = G.nodes[u]["lon"]
                lat2 = G.nodes[v]["lat"]
                lon2 = G.nodes[v]["lon"]
                dist = _haversine(lat1, lon1, lat2, lon2)
                # añadimos arista si no existe o si encontramos una más corta
                if not G.has_edge(u, v) or dist < G[u][v].get("weight", float("inf")):
                    G.add_edge(u, v, weight=dist, length=dist)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_graph_from_file(file_path: str) -> nx.Graph:
    """
    Carga el grafo desde un archivo .osm y lo deja en memoria.
    Lo usa el endpoint /upload-graph en main.py.
    """
    global _GRAPH
    _GRAPH = _parse_osm_to_graph(file_path)
    return _GRAPH


def get_graph() -> nx.Graph:
    """
    Devuelve el grafo en memoria. Lanza error si aún no se ha cargado.
    """
    if _GRAPH is None:
        raise ValueError("Graph not loaded. Call load_graph_from_file() first.")
    return _GRAPH


def get_graph_data() -> Dict[str, Any]:
    """
    Devuelve una versión serializable del grafo para el front:
    {
      "nodes": [{id, lat, lon}, ...],
      "edges": [{from, to, weight}, ...]
    }
    """
    G = get_graph()

    nodes = [
        {"id": int(n), "lat": float(data["lat"]), "lon": float(data["lon"])}
        for n, data in G.nodes(data=True)
    ]

    edges = [
        {"from": int(u), "to": int(v), "weight": float(data.get("weight", 1.0))}
        for u, v, data in G.edges(data=True)
    ]

    return {"nodes": nodes, "edges": edges}
