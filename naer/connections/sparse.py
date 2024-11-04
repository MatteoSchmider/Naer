from naer.connections.connection import Connection, naer_rand


@Connection
class sparse:
    def __init__(self, probability) -> None:
        self.probability: float = probability

    def connect(self, from_addr: int, to_addr: int) -> bool:
        if naer_rand() <= self.probability:
            return True
        return False
