import configparser

MEGABYTE_SIZE = 1024 * 1024


class ConfigReader:

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, config_path: str = "./config/config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        pass

    def read_str(self, section: str, key: str) -> str:
        return self.config.get(section, key)

    def read_int(self, section: str, key: str) -> int:
        return self.config.getint(section, key)

    def read_float(self, section: str, key: str) -> float:
        return self.config.getfloat(section, key)

    def read_all_dirs(self, section: str) -> list[str]:
        dir_list = []
        for key in self.config.options(section):
            dir_list.append(self.config.get(section, key))
        return dir_list

    def read_bytes_chunk_size(self) -> float:
        m_bytes_chunk_size = self.read_float("grpc", "MAX_CHUNK_SIZE_MB")
        bytes_chunk_size = int(m_bytes_chunk_size * MEGABYTE_SIZE)

        return bytes_chunk_size
