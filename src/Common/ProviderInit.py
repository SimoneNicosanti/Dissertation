import subprocess

import onnxruntime as ort


def init_providers_list() -> list[str]:
    providers = []
    if (
        "CUDAExecutionProvider" in ort.get_available_providers()
        and test_gpu_available()
    ):
        providers.append("CUDAExecutionProvider")

    elif "OpenVINOExecutionProvider" in ort.get_available_providers():
        providers.append("OpenVINOExecutionProvider")

    providers.append("CPUExecutionProvider")

    return providers


def test_gpu_available() -> bool:

    available = False
    try:
        # trunk-ignore(bandit/B603)
        # trunk-ignore(bandit/B607)
        return_code = subprocess.run(
            ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL
        ).returncode
        if return_code == 0:
            available = True

    except Exception as _:
        available = False

    return available


def test_cuda_ep(providers: list[str]) -> bool:
    return "CUDAExecutionProvider" in providers


def test_openvino_ep(providers: list[str]) -> bool:
    return "OpenVINOExecutionProvider" in providers
