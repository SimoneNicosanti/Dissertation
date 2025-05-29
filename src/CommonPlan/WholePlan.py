from CommonPlan.Plan import Plan


class WholePlan:
    def __init__(self):
        self.plan_dict: dict[str, Plan] = {}
        ## Add other info about the prodeced plan
        ## Example the costs of latencies and energy

    def put_model_plan(self, model_name: str, plan: Plan):
        self.plan_dict[model_name] = plan

    def get_model_plan(self, model_name: str) -> Plan:
        return self.plan_dict[model_name]

    def get_model_names(self) -> list[str]:
        return list(self.plan_dict.keys())

    def encode(self):
        whole_plan_dict = {}
        for model_name, model_plan in self.plan_dict.items():
            whole_plan_dict[model_name] = model_plan.encode()
        return whole_plan_dict

    @staticmethod
    def decode(whole_plan_dict: dict) -> "WholePlan":
        whole_plan = WholePlan()
        for model_name, model_plan_dict in whole_plan_dict.items():
            plan = Plan.decode(model_plan_dict)
            whole_plan.put_model_plan(model_name, plan)
        return whole_plan
