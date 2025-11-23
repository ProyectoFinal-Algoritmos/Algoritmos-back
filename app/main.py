# app/main.py

from fastapi import FastAPI

app = FastAPI(
    title="TSP Road Network API",
    version="1.0.0",
    description="Backend para cargar red vial, puntos y ejecutar algoritmos de TSP."
)


@app.get("/health")
def health_check():
    """
    Endpoint simple para verificar que el backend está levantado.
    """
    return {"status": "ok"}
