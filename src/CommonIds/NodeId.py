from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class NodeId:
    node_name: str

    def __post_init__(self):
        # Force conversion to string, regardless of input type
        object.__setattr__(self, "node_name", str(self.node_name))

    def __repr__(self):
        return self.node_name

    def encode(self):
        return self.node_name

    @staticmethod
    def decode(node_name: str) -> "NodeId":
        return NodeId(node_name)
