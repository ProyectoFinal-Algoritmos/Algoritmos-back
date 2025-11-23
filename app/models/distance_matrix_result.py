from dataclasses import dataclass
from typing import List

@dataclass
class DistanceMatrixResult:
    # representa distancia desde nodo i hasta nodo j
    distances: List[List[float]]
    # representa la ruta con la menor distancia para llegar desde i hasta j
    paths: List[List[List[int]]]

