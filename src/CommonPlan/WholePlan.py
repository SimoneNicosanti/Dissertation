from CommonIds.NodeId import NodeId
from CommonPlan.Plan import Plan


class WholePlan:
    def __init__(self, start_server: NodeId):
        self.plan_dict: dict[str, Plan] = {}
        self.start_server: NodeId = start_server
        ## Add other info about the prodeced plan
        ## Example the costs of latencies and energy

    def put_model_plan(self, model_name: str, plan: Plan):
        self.plan_dict[model_name] = plan

    def get_model_plan(self, model_name: str) -> Plan:
        return self.plan_dict[model_name]

    def get_model_names(self) -> list[str]:
        return list(self.plan_dict.keys())

    def get_start_server(self) -> str:
        return self.start_server.node_name

    def encode(self):
        whole_plan_dict = {}
        whole_plan_dict["start_server"] = self.start_server.node_name
        whole_plan_dict["plan_dict"] = {}
        for model_name, model_plan in self.plan_dict.items():
            whole_plan_dict["plan_dict"][model_name] = model_plan.encode()
        return whole_plan_dict

    @staticmethod
    def decode(whole_plan_dict: dict) -> "WholePlan":
        start_server = NodeId(whole_plan_dict["start_server"])
        whole_plan = WholePlan(start_server)
        for model_name, model_plan_dict in whole_plan_dict["plan_dict"].items():
            plan = Plan.decode(model_plan_dict)
            whole_plan.put_model_plan(model_name, plan)
        return whole_plan
