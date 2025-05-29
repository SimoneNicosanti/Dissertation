from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class NodeId:
    node_name: str

    def __repr__(self):
        return self.node_name

    def encode(self):
        return self.node_name

    @staticmethod
    def decode(node_name: str) -> "NodeId":
        return NodeId(node_name)