from dataclasses import dataclass


@dataclass
class Spike:
    time: int
    address: int
    trace: float
    trace_time: int


@dataclass
class Synapse:
    weight: float
