from CommonIds.NodeId import NodeId


class ModelExecutionProfile:

    def __init__(self):
        self.model_execution_profile_dict: dict[NodeId, list[float]] = {}

    def encode(self) -> dict:
        transformed_dict = {}
        for node_id in self.model_execution_profile_dict:
            transformed_dict[node_id.node_name] = self.model_execution_profile_dict[
                node_id
            ]
        return transformed_dict

    @staticmethod
    def decode(transformed_dict: dict) -> "ModelExecutionProfile":
        model_execution_profile = ModelExecutionProfile()

        for node_name, layer_execution_profile in transformed_dict.items():
            model_execution_profile.put_layer_execution_profile(
                NodeId(node_name),
                layer_execution_profile["nq_avg_time"],
                layer_execution_profile["nq_med_time"],
                False,
            )
            model_execution_profile.put_layer_execution_profile(
                NodeId(node_name),
                layer_execution_profile["q_avg_time"],
                layer_execution_profile["q_med_time"],
                True,
            )

        return model_execution_profile

    def put_layer_execution_profile(
        self,
        node_id: NodeId,
        avg_exec_time: float,
        med_exec_time: float,
        is_quantized: bool,
    ):

        self.model_execution_profile_dict.setdefault(node_id, {})
        if not is_quantized:
            self.model_execution_profile_dict[node_id]["nq_avg_time"] = avg_exec_time
            self.model_execution_profile_dict[node_id]["nq_med_time"] = med_exec_time
        else:
            self.model_execution_profile_dict[node_id]["q_avg_time"] = avg_exec_time
            self.model_execution_profile_dict[node_id]["q_med_time"] = med_exec_time

    def get_not_quantized_layer_time(self, node_id: NodeId) -> float:
        if node_id not in self.model_execution_profile_dict:
            return 0

        pred_layer_time = self.model_execution_profile_dict[node_id]["nq_avg_time"]
        pred_tot_time = self.get_total_not_quantized_time()
        real_tot_time = self.model_execution_profile_dict[NodeId("WholeModel")][
            "nq_avg_time"
        ]
        return (pred_layer_time / pred_tot_time) * real_tot_time

    def get_quantized_layer_time(self, node_id: NodeId) -> float:
        if node_id not in self.model_execution_profile_dict:
            return 0

        pred_layer_time = self.model_execution_profile_dict[node_id]["q_avg_time"]
        pred_tot_time = self.get_total_quantized_time()
        real_tot_time = self.model_execution_profile_dict[NodeId("WholeModel")][
            "q_avg_time"
        ]
        return (pred_layer_time / pred_tot_time) * real_tot_time

    def get_total_not_quantized_time(self) -> float:
        if NodeId("TotalSum") in self.model_execution_profile_dict:
            return self.model_execution_profile_dict[NodeId("TotalSum")]["nq_avg_time"]

        tot_avg_time = 0
        for node_id in self.model_execution_profile_dict:
            if node_id.node_name == "WholeModel" or node_id.node_name == "TotalSum":
                continue
            tot_avg_time += self.model_execution_profile_dict[node_id]["nq_avg_time"]
        return tot_avg_time

    def get_total_quantized_time(self) -> float:
        if NodeId("TotalSum") in self.model_execution_profile_dict:
            return self.model_execution_profile_dict[NodeId("TotalSum")]["q_avg_time"]

        tot_avg_time = 0
        for node_id in self.model_execution_profile_dict:
            if node_id.node_name == "WholeModel" or node_id.node_name == "TotalSum":
                continue
            tot_avg_time += self.model_execution_profile_dict[node_id]["q_avg_time"]
        return tot_avg_time


class ServerExecutionProfile:

    def __init__(self):

        ## Maps model_name --> execution_profile of the model for a server
        self.execution_profile_dict: dict[str, ModelExecutionProfile] = {}

    def put_model_execution_profile(
        self, model_name: str, model_execution_profile: ModelExecutionProfile
    ):
        self.execution_profile_dict[model_name] = model_execution_profile

        pass

    def get_model_execution_profile(self, model_name: str) -> ModelExecutionProfile:
        return self.execution_profile_dict[model_name]

    def encode(self) -> dict:
        transformed_dict = {}
        for model_name in self.execution_profile_dict:
            transformed_dict[model_name] = self.execution_profile_dict[
                model_name
            ].encode()
        return transformed_dict

    @staticmethod
    def decode(transformed_dict: dict) -> "ServerExecutionProfile":
        server_execution_profile = ServerExecutionProfile()
        for model_name, model_exec_profile_dict in transformed_dict.items():
            server_execution_profile.put_model_execution_profile(
                model_name, ModelExecutionProfile.decode(model_exec_profile_dict)
            )

        return server_execution_profile


class ServerExecutionProfilePool:

    def __init__(self):
        self.execution_profile_pool_dict: dict[NodeId, ServerExecutionProfile] = {}

    def put_execution_profiles_for_server(
        self, node_id: NodeId, execution_profile: ServerExecutionProfile
    ):
        self.execution_profile_pool_dict[node_id] = execution_profile

    def get_execution_profiles_for_server(
        self, node_id: NodeId
    ) -> ServerExecutionProfile:
        return self.execution_profile_pool_dict[node_id]

    def encode(self) -> dict:
        transformed_dict = {}
        for node_id in self.execution_profile_pool_dict:
            transformed_dict[node_id.node_name] = self.execution_profile_pool_dict[
                node_id
            ].encode()
        return transformed_dict

    @staticmethod
    def decode(transformed_dict: dict) -> "ServerExecutionProfilePool":
        server_execution_profile_pool = ServerExecutionProfilePool()
        for node_name, execution_profile_json in transformed_dict.items():
            server_execution_profile_pool.put_execution_profiles_for_server(
                NodeId(node_name),
                ServerExecutionProfile.decode(execution_profile_json),
            )

        return server_execution_profile_pool
