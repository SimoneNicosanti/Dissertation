import multiprocessing.connection
import sys
import onnxruntime as ort
import numpy as np
import multiprocessing
import time
import threading

# Carica il modello
model_path = "../src/Other/models/yolo11l.onnx"

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
# Crea due sessioni separate
session1 = ort.InferenceSession(model_path, sess_options=sess_options)

# Input di test
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
input_name = session1.get_inputs()[0].name
output_name = session1.get_outputs()[0].name

# Funzione per eseguire l'inferenza
def run_inference(session, input_data, input_name, output_name, queue):
    result = session.run([output_name], {input_name: input_data})


def worker_process(child_conn : multiprocessing.connection.Connection, model_path): 
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, sess_options=sess_options)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    while True:
        input_name = child_conn.recv()
        out = session.run(None, {input_name: input_data})
        child_conn.send(None)

# Test sequenziale
start = time.time()
for i in range(30):
    run_inference(session1, input_data, input_name, output_name, None)
end = time.time()
print(f"Esecuzione sequenziale: {end - start:.4f} sec")

# Test con multi-processo

parent_conn_1, child_conn_1 = multiprocessing.Pipe()
process1 = multiprocessing.Process(target=worker_process, args=(child_conn_1, model_path))

parent_conn_2, child_conn_2 = multiprocessing.Pipe()
process2 = multiprocessing.Process(target=worker_process, args=(child_conn_2, model_path))

process1.start()
process2.start()
start = time.time()
for i in range(15):
    parent_conn_1.send((input_name))
    parent_conn_2.send((input_name))

for i in range(15):
    parent_conn_1.recv()
    parent_conn_2.recv()

end = time.time()
print(f"Esecuzione con processi: {end - start:.4f} sec")

session2 = ort.InferenceSession(model_path, sess_options=sess_options)
start = time.time()
for i in range(15):
    thr_1 = threading.Thread(target=run_inference, args=(session1, input_data, input_name, output_name, None))
    thr_2 = threading.Thread(target=run_inference, args=(session2, input_data, input_name, output_name, None))

    thr_1.start()
    thr_2.start()

    thr_1.join()
    thr_2.join()
end = time.time()
print(f"Esecuzione con thread: {end - start:.4f} sec")