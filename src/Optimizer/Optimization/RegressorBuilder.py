import networkx as nx
import pulp

from CommonProfile.ModelProfile import Regressor
from Optimizer.Optimization.OptimizationKeys import QuantizationKey


class RegressorBuilder:

    @staticmethod
    def build_regressor_expression(
        problem: pulp.LpProblem,
        regressor: Regressor,
        model_graph: nx.MultiDiGraph,
        quant_vars: dict[QuantizationKey, pulp.LpVariable],
    ) -> pulp.LpAffineExpression:

        model_name = model_graph.graph["name"]
        ## Set up var for existence
        exist_var = pulp.LpVariable("exist", cat=pulp.LpBinary)

        model_quant_keys = filter(
            lambda x: x.mod_name == model_name,
            quant_vars.keys(),
        )
        sum_vars = 0
        for quant_key in model_quant_keys:
            quant_var = quant_vars[quant_key]
            sum_vars += quant_var
            problem += exist_var >= quant_var
        problem += exist_var <= sum_vars

        ## Set up regressor encoding as linear model
        regressor_expression = regressor.get_intercept() * exist_var

        idx = 0
        for interaction_key, interaction_value in regressor.get_interactions().items():
            var_name = f"reg_prod_{model_name}_{idx}"
            prod_var = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            quant_sum_vars = 0
            for node_id in interaction_key:
                quant_key = QuantizationKey(node_id, model_name)
                quant_var = quant_vars[quant_key]
                problem += prod_var <= quant_var

                quant_sum_vars += quant_var

            problem += prod_var <= quant_sum_vars

            regressor_expression += interaction_value * prod_var

            idx += 1

        return regressor_expression
