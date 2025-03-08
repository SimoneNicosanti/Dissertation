import numpy


class OutputSender:
    def __init__(self):
        pass

    def send_output(
        self,
        request_info: RequestInfo,
        output_dict: dict[str, numpy.ndarray],
        plan: dict[str],
    ):
        ## TODO
        ## for each output
        ## check plan for next component id
        ## send to next component handler
        ## Find next handler using registry
        ## Cache this connections in order not to recreate it every time

        pass
