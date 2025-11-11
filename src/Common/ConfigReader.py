import configparser
import threading

MEGABYTE_SIZE = 1024 * 1024


class ConfigReader:

    __instance = None
    __init_lock = threading.Lock()

    def __new__(cls, config_path: str = "./config/config.ini"):
        if cls.__instance is None:
            with cls.__init_lock:
                if cls.__instance is None:
                    instance = super().__new__(cls)
                    instance.__initialize(config_path)
                    cls.__instance = instance
        return cls.__instance

    def __initialize(self, config_path: str):
        self.config_path = config_path  # opzionale
        config = configparser.ConfigParser()
        read_files = config.read(config_path)
        if not read_files:
            raise FileNotFoundError(f"Could not read config file at {config_path}")
        self.config = config

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

    def read_bytes_chunk_size(self) -> int:
        m_bytes_chunk_size = self.read_float("grpc", "MAX_CHUNK_SIZE_MB")
        bytes_chunk_size = int(m_bytes_chunk_size * MEGABYTE_SIZE)

        return bytes_chunk_size
