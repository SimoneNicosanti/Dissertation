import multiprocessing
import multiprocessing.connection
import threading

from CommonServer.InferenceInfo import ComponentInfo, SharedTensorInfo


class PipeWrapper:

    def __init__(self, master_conn: multiprocessing.connection.Connection):
        self.master_conn = master_conn
        self.lock = threading.Lock()

    def call_inference(
        self, component_info: ComponentInfo, input_list: list[SharedTensorInfo]
    ) -> list[SharedTensorInfo]:
        ## Only one can call the inference at a time
        with self.lock:
            send_value = (component_info, input_list)
            self.master_conn.send(send_value)
            output_list = self.master_conn.recv()

            return output_list
