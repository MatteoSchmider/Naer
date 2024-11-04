from naer.connections.connection import Connection


@Connection
class full:
    def connect(self, from_addr: int, to_addr: int) -> bool:
        return True
