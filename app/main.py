from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
from typing import List, Tuple

from app.services.graph_loader import get_graph, get_graph_data, load_graph_from_file
from app.services.process_nodes import (
    process_points_into_graph,
    load_points_from_uploaded_file,
    get_selected_nodes,
)
from app.services.distance_matrix import (
    build_distance_matrix_with_paths,
    get_distance_matrix,
    set_distance_matrix,
)
from app.services.tsp_solver import (
    solve_tsp_brute_force,
    solve_tsp_greedy,
    solve_tsp_dynamic_programming,
)
from app.utils.path_utils import map_path_indices_to_ids, reconstruct_full_path


app = FastAPI()

# CORS para permitir peticiones desde el front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-graph")
async def upload_graph(file: UploadFile = File(...)):
    if not file.filename.endswith(".osm"):
        raise HTTPException(status_code=400, detail="Only .osm files are supported")

    try:
        result = load_graph_from_file(file.file)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")


@app.get("/graph-data")
def graph_data():
    try:
        return get_graph_data()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload-points")
async def upload_points(file: UploadFile = File(...)):
    try:
        G = get_graph()
        node_ids = load_points_from_uploaded_file(file.file, G)
        return {
            "status": "success",
            "numPoints": len(node_ids),
            "nodeIds": node_ids,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/build-matrix")
def build_matrix():
    try:
        G = get_graph()
        node_ids = get_selected_nodes()

        if not node_ids or len(node_ids) < 2:
            raise HTTPException(
                status_code=400, detail="At least 2 points are required."
            )

        matrix = build_distance_matrix_with_paths(G, node_ids)
        set_distance_matrix(matrix)

        return {
            "status": "success",
            "numPoints": len(node_ids),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# solución de TSP para programación dinámica (Held-Karp)
@app.get("/tsp/dynamic")
def run_held_karp():
    try:
        matrix = get_distance_matrix()
        node_ids = get_selected_nodes()

        if not matrix or not node_ids:
            raise HTTPException(
                status_code=400, detail="Missing matrix or selected nodes."
            )

        result = solve_tsp_dynamic_programming(matrix.distances)

        real_path = map_path_indices_to_ids(result.path, node_ids)
        full_path = reconstruct_full_path(result.path, matrix.paths)

        return {
            "status": "success",
            "result": {
                "algorithmName": result.algorithmName,
                "path": real_path,  # para estadísticas
                "total_cost": result.total_cost,
                "execution_time": result.execution_time,
            },
            "fullPath": full_path,  # para dibujar en el mapa
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# fuerza bruta
@app.get("/tsp/brute-force")
def run_brute_force():
    try:
        matrix = get_distance_matrix()
        node_ids = get_selected_nodes()

        if not matrix or not node_ids:
            raise HTTPException(
                status_code=400, detail="Missing matrix or selected nodes."
            )

        result = solve_tsp_brute_force(matrix.distances)

        real_path = map_path_indices_to_ids(result.path, node_ids)
        full_path = reconstruct_full_path(result.path, matrix.paths)

        return {
            "status": "success",
            "result": {
                "algorithmName": result.algorithmName,
                "path": real_path,
                "total_cost": result.total_cost,
                "execution_time": result.execution_time,
            },
            "fullPath": full_path,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Greedy (vecino más cercano)
@app.get("/tsp/greedy")
def run_greedy():
    try:
        matrix = get_distance_matrix()
        node_ids = get_selected_nodes()

        if not matrix or not node_ids:
            raise HTTPException(
                status_code=400, detail="Missing matrix or selected nodes."
            )

        result = solve_tsp_greedy(matrix.distances)

        real_path = map_path_indices_to_ids(result.path, node_ids)
        full_path = reconstruct_full_path(result.path, matrix.paths)

        return {
            "status": "success",
            "result": {
                "algorithmName": result.algorithmName,
                "path": real_path,
                "total_cost": result.total_cost,
                "execution_time": result.execution_time,
            },
            "fullPath": full_path,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# FUNCIONES DE PRUEBA POR CONSOLA
# ---------------------------

def load_points_to_visit(file_path: str) -> List[Tuple[float, float]]:
    """
    Lee un archivo data/points.txt donde cada línea es:
    lat lon
    y devuelve una lista de tuplas (lat, lon).
    """
    points: List[Tuple[float, float]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            lat = float(parts[0])
            lon = float(parts[1])
            points.append((lat, lon))
    return points


def main():
    # 1. Cargar grafo desde archivo .osm local
    osm_path = "data/chapinero.osm"  # ajusta el nombre de archivo si es otro
    with open(osm_path, "rb") as f:
        load_graph_from_file(f)

    G = get_graph()
    print(f"Grafo inicial: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    # 2. Cargar puntos desde archivo txt local
    file_path = "data/points.txt"  # ajusta si tu archivo se llama diferente
    points = load_points_to_visit(file_path)
    print(f"Leídos {len(points)} puntos desde archivo.")

    # 3. Procesar nuevos nodos leídos (insertarlos en el grafo si no existen)
    final_node_ids = process_points_into_graph(G, points)
    print("Nodos a visitar (IDs en el grafo):", final_node_ids)

    print(
        f"Grafo actualizado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas"
    )

    # 4. Crear matriz de distancias según los nodos que se quieren visitar
    result_matrix = build_distance_matrix_with_paths(G, final_node_ids)

    print("\nMatriz de distancias:")
    for row in result_matrix.distances:
        pretty = [
            "{:.1f}".format(val) if val != float("inf") else "∞"
            for val in row
        ]
        print(pretty)

    print("\nMatriz de caminos reales (solo i != j):")
    for i in range(len(result_matrix.paths)):
        for j in range(len(result_matrix.paths)):
            if i != j:
                print(
                    f"De {final_node_ids[i]} a {final_node_ids[j]}: {result_matrix.paths[i][j]}"
                )

    # 5. Ejecutar TSP por fuerza bruta
    result_brute_force = solve_tsp_brute_force(result_matrix.distances)

    print("\n=== Fuerza bruta ===")
    print("Nombre del algoritmo usado:")
    print(result_brute_force.algorithmName)
    print("Ruta óptima (índices en matriz):")
    print(result_brute_force.path)
    print(f"Costo total: {result_brute_force.total_cost:.2f} metros")
    print(f"Tiempo de ejecución: {result_brute_force.execution_time:.4f} segundos")

    id_path = map_path_indices_to_ids(result_brute_force.path, final_node_ids)
    print("Ruta como IDs reales:")
    print(id_path)

    full_real_path = reconstruct_full_path(
        result_brute_force.path, result_matrix.paths
    )
    print("Ruta completa con nodos intermedios incluidos:")
    print(full_real_path)

    # 6. Ejecutar el algoritmo de programación dinámica (Held-Karp)
    result_dynamic = solve_tsp_dynamic_programming(result_matrix.distances)

    print("\n=== Programación Dinámica (Held-Karp) ===")
    print("Nombre del algoritmo usado:")
    print(result_dynamic.algorithmName)
    print("Ruta óptima (índices en matriz):")
    print(result_dynamic.path)
    print(f"Costo total: {result_dynamic.total_cost:.2f} metros")
    print(f"Tiempo de ejecución: {result_dynamic.execution_time:.4f} segundos")

    id_path_dynamic = map_path_indices_to_ids(result_dynamic.path, final_node_ids)
    print("Ruta como IDs reales:")
    print(id_path_dynamic)

    full_real_path_dynamic = reconstruct_full_path(
        result_dynamic.path, result_matrix.paths
    )
    print("Ruta completa con nodos intermedios incluidos:")
    print(full_real_path_dynamic)


if __name__ == "__main__":
    main()
