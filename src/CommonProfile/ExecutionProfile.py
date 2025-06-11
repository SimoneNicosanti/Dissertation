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
                NodeId(node_name), layer_execution_profile[0], False
            )
            model_execution_profile.put_layer_execution_profile(
                NodeId(node_name), layer_execution_profile[1], True
            )

        return model_execution_profile

    def put_layer_execution_profile(
        self, node_id: NodeId, execution_time: float, is_quantized: bool
    ):
        self.model_execution_profile_dict.setdefault(node_id, [0, 0])
        idx = 1 if is_quantized else 0
        self.model_execution_profile_dict[node_id][idx] = execution_time

    def get_not_quantized_layer_time(self, node_id: NodeId) -> float:
        if node_id not in self.model_execution_profile_dict:
            return 0
        return self.model_execution_profile_dict[node_id][0]

    def get_quantized_layer_time(self, node_id: NodeId) -> float:
        if node_id not in self.model_execution_profile_dict:
            return 0
        return self.model_execution_profile_dict[node_id][1]


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
