from dataclasses import dataclass
from typing import List

@dataclass
class TSPResult:
    path: List[int]
    total_cost: float
    execution_time: float  # en segundos 
    algorithmName: str
