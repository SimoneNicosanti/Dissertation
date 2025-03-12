import multiprocessing
import multiprocessing.connection
from multiprocessing import shared_memory

import numpy
import onnxruntime as ort

from CommonServer.InferenceInfo import ComponentInfo, SharedTensorInfo


def prepare_input(input_list: list[SharedTensorInfo]) -> dict[str, numpy.ndarray]:

    input_dict = {}
    shared_mem_refs = []
    for shared_tensor_info in input_list:
        shared_mem_name = shared_tensor_info.shared_memory_name
        shared_mem = shared_memory.SharedMemory(shared_mem_name, create=False)

        input_tensor = numpy.ndarray(
            buffer=shared_mem.buf,
            dtype=shared_tensor_info.tensor_type,
            shape=shared_tensor_info.tensor_shape,
        )

        input_dict[shared_tensor_info.tensor_name] = input_tensor

        shared_mem_refs.append(shared_mem)

    return input_dict, shared_mem_refs


def prepare_output(output_dict: dict[str, numpy.ndarray]) -> list[SharedTensorInfo]:

    shared_output_tensor_list = []
    for output_name, output_tensor in output_dict.items():
        shared_mem = shared_memory.SharedMemory(create=True, size=output_tensor.nbytes)

        shared_mem.buf[:] = output_tensor[:]

        shared_tensor_info = SharedTensorInfo(
            tensor_name=output_name,
            tensor_type=output_tensor.dtype,
            tensor_shape=output_tensor.shape,
            shared_memory_name=shared_mem.name,
        )

        shared_output_tensor_list.append(shared_tensor_info)

        shared_mem.close()

    return shared_output_tensor_list


def do_inference(
    infer_session: ort.InferenceSession, input_dict: dict[str, numpy.ndarray]
):
    output_names = [out.name for out in infer_session.get_outputs()]
    output_list = infer_session.run(output_names=output_names, input_feed=input_dict)
    output_dict = dict(zip(output_names, output_list))

    return output_dict


def close_shared_memory(shared_mem_refs: list[shared_memory.SharedMemory]):
    for shared_mem in shared_mem_refs:
        shared_mem.close()


def return_result(self, output_list: list[SharedTensorInfo]):
    self.worker_conn.send(output_list)


def receive_input(self) -> tuple[ComponentInfo, list[SharedTensorInfo]]:
    return self.worker_conn.recv()


def start_worker_process(
    components_dict: dict[ComponentInfo, str],
    worker_conn: multiprocessing.connection.Connection,
) -> None:

    inference_session_dict: dict[ComponentInfo, ort.InferenceSession] = {}

    for component_info, component_path in components_dict.items():
        inference_session_dict[component_info] = ort.InferenceSession(component_path)

    while input_data := worker_conn.recv():
        comp_info, shared_input_tensor_list = input_data

        input_dict, shared_mem_refs = prepare_input(shared_input_tensor_list)

        output_dict = do_inference(
            infer_session=inference_session_dict[comp_info], input_dict=input_dict
        )

        close_shared_memory(shared_mem_refs)

        shared_output_tensor_list = prepare_output(output_dict=output_dict)

        worker_conn.send(shared_output_tensor_list)
