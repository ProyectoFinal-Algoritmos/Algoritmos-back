
import io
import os

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app


DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")


# ---------- FIXTURE CLIENTE ----------

@pytest.fixture
def client():
    return TestClient(app)


# ---------- /upload-graph ----------

def test_upload_graph_extension_invalida(client):
    """Debe rechazar archivos que no sean .osm"""
    fake_file = io.BytesIO(b"no importa")
    resp = client.post(
        "/upload-graph",
        files={"file": ("points.txt", fake_file, "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Only .osm files are supported"


def test_upload_graph_ok(client, monkeypatch):
    """
    Debe aceptar un .osm y devolver lo que retorne load_graph_from_file.
    No dependemos de la implementación real: se parchea.
    """
    # contenido arbitrario; como vamos a parchear, no importa
    fake_file = io.BytesIO(b"<osm></osm>")

    # lo que queremos que devuelva el backend
    fake_result = {"nodes": 123, "edges": 456}

    # parchear la función usada por el endpoint
    monkeypatch.setattr(main, "load_graph_from_file", lambda f: fake_result)

    resp = client.post(
        "/upload-graph",
        files={"file": ("chapinero.osm", fake_file, "application/xml")},
    )

    assert resp.status_code == 200
    assert resp.json() == fake_result


# ---------- /graph-data ----------

def test_graph_data_success(client, monkeypatch):
    esperado = {"nodes": [1, 2, 3], "edges": []}
    monkeypatch.setattr(main, "get_graph_data", lambda: esperado)

    resp = client.get("/graph-data")
    assert resp.status_code == 200
    assert resp.json() == esperado


def test_graph_data_error(client, monkeypatch):
    # forzamos que get_graph_data lance una excepción
    def boom():
        raise Exception("fail")

    monkeypatch.setattr(main, "get_graph_data", boom)

    resp = client.get("/graph-data")
    assert resp.status_code == 400
    assert "fail" in resp.json()["detail"]


# ---------- FIXTURE GLOBAL PARA TSP (por defecto) ----------

@pytest.fixture(autouse=True)
def setup_tsp(monkeypatch):
    """
    Parcheo por defecto para las pruebas de TSP, salvo cuando
    explícitamente se sobreescriba en un test.
    """

    # matriz falsa
    class FakeMatrix:
        distances = [[0, 1], [1, 0]]
        paths = [[[], [1]], [[0], []]]

    monkeypatch.setattr(main, "get_distance_matrix", lambda: FakeMatrix())
    monkeypatch.setattr(main, "get_selected_nodes", lambda: [10, 20])

    # resultado falso de los tres algoritmos
    class Resultado:
        algorithmName = "dummy"
        path = [0, 1]
        total_cost = 1.23
        execution_time = 0.004

    monkeypatch.setattr(main, "solve_tsp_brute_force", lambda d: Resultado())
    monkeypatch.setattr(main, "solve_tsp_greedy", lambda d: Resultado())
    monkeypatch.setattr(main, "solve_tsp_dynamic_programming", lambda d: Resultado())

    # no nos importa el camino completo para estas pruebas
    monkeypatch.setattr(main, "reconstruct_full_path", lambda path, paths: [])


# ---------- /upload-points ----------

@pytest.mark.parametrize(
    "returned_nodes",
    [
        [],          # 0 nodos
        [42],        # 1 nodo
        [1, 2, 3, 4, 5],  # 5 nodos
    ],
)
def test_upload_points_varias_cantidades(client, monkeypatch, returned_nodes):
    """
    /upload-points debe devolver numPoints y nodeIds coherentes con
    lo que retorne load_points_from_uploaded_file.
    """
    # no usamos el grafo real
    monkeypatch.setattr(main, "get_graph", lambda: None)

    # simulamos la lectura de puntos
    monkeypatch.setattr(
        main,
        "load_points_from_uploaded_file",
        lambda f, G: returned_nodes,
    )

    resp = client.post(
        "/upload-points",
        files={"file": ("dummy.txt", io.BytesIO(b""), "text/plain")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["numPoints"] == len(returned_nodes)
    assert body["nodeIds"] == returned_nodes


# ---------- /build-matrix ----------

@pytest.mark.parametrize(
    "selected_nodes, expected_status",
    [
        ([], 500),      # < 2 nodos → actualmente termina en 500 (wrap del HTTPException 400)
        ([1], 500),
        ([1, 2], 200),  # 2 nodos → OK
        ([1, 2, 3], 200),
    ],
)
def test_build_matrix_varias_cantidades(
    client, monkeypatch, selected_nodes, expected_status
):
    # no usamos el grafo real
    monkeypatch.setattr(main, "get_graph", lambda: None)
    monkeypatch.setattr(main, "get_selected_nodes", lambda: selected_nodes)

    if expected_status == 200:
        # solo hace falta que devuelva "algo" y que set_distance_matrix lo acepte
        monkeypatch.setattr(
            main, "build_distance_matrix_with_paths", lambda G, ns: "fake-matrix"
        )
        monkeypatch.setattr(main, "set_distance_matrix", lambda m: None)

    resp = client.get("/build-matrix")
    assert resp.status_code == expected_status

    if expected_status == 200:
        body = resp.json()
        assert body["status"] == "success"
        assert body["numPoints"] == len(selected_nodes)
    else:
        # el detalle debería contener el mensaje original
        assert "At least 2 points are required" in resp.json()["detail"]


# ---------- TSP: casos simples por endpoint ----------

def test_tsp_brute_force(client):
    resp = client.get("/tsp/brute-force")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    # path [0,1] debe mapear a IDs [10,20]
    assert body["result"]["path"] == [10, 20]


def test_tsp_greedy(client):
    resp = client.get("/tsp/greedy")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_tsp_dynamic(client):
    resp = client.get("/tsp/dynamic")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


# ---------- TSP con 2, 3 y 4 nodos en los 3 endpoints ----------

@pytest.mark.parametrize(
    "nodes, fake_path, expected_mapped",
    [
        ([10, 20], [0, 1], [10, 20]),
        ([3, 5, 7], [2, 0, 1], [7, 3, 5]),
        ([100, 200, 300, 400], [3, 1, 2, 0], [400, 200, 300, 100]),
    ],
)
@pytest.mark.parametrize("endpoint", ["/tsp/brute-force", "/tsp/greedy", "/tsp/dynamic"])
def test_tsp_varias_cantidades(
    client, monkeypatch, nodes, fake_path, expected_mapped, endpoint
):
    """
    Verifica que el mapeo de índices de la matriz → IDs reales funcione
    para los tres endpoints de TSP.
    """
    class FakeMatrix:
        distances = []
        paths = []

    monkeypatch.setattr(main, "get_distance_matrix", lambda: FakeMatrix())
    monkeypatch.setattr(main, "get_selected_nodes", lambda: nodes)

    class Resultado:
        algorithmName = "dummy"
        path = fake_path
        total_cost = 99.9
        execution_time = 0.123

    if "brute-force" in endpoint:
        monkeypatch.setattr(main, "solve_tsp_brute_force", lambda d: Resultado())
    elif "greedy" in endpoint:
        monkeypatch.setattr(main, "solve_tsp_greedy", lambda d: Resultado())
    else:
        monkeypatch.setattr(main, "solve_tsp_dynamic_programming", lambda d: Resultado())

    # no nos interesa el fullPath aquí
    monkeypatch.setattr(main, "reconstruct_full_path", lambda path, paths: [])

    resp = client.get(endpoint)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["result"]["path"] == expected_mapped
