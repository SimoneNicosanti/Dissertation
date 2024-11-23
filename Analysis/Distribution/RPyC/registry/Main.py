import rpyc
from rpyc.utils.server import ThreadedServer


class RegisterService(rpyc.Service):
    layerMap = {}

    def __init__(self):
        super().__init__()

    def exposed_subscribeLayer(self, layerName: str, host: str, port: int):
        print(f"Subscribed {layerName} on {host}:{port}")
        RegisterService.layerMap[layerName] = (host, port)

    def exposed_getLayerHost(self, layerName: str):
        print(f"Called for {layerName}")
        return RegisterService.layerMap[layerName]


if __name__ == "__main__":
    t = ThreadedServer(RegisterService, port="8000")
    print("Starting Service")
    t.start()
