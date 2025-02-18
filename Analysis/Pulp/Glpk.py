"""
The Simplified Whiskas Model Python Formulation for the PuLP Modeller

Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""

# Import PuLP modeler functions
import pulp
from pulp import constants

solver_list = pulp.listSolvers(onlyAvailable=True)
print(solver_list)

# Create the 'prob' variable to contain the problem data
prob = pulp.LpProblem("The Whiskas Problem", constants.LpMinimize)

# The 2 variables Beef and Chicken are created with a lower limit of zero
x1 = pulp.LpVariable("ChickenPercent", 0, None, constants.LpInteger)
x2 = pulp.LpVariable("BeefPercent", 0)

# The objective function is added to 'prob' first
prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"

# The five constraints are entered
prob += x1 + x2 == 100, "PercentagesSum"
prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

# The problem data is written to an .lp file
prob.writeLP("WhiskasModel.lp")

# The problem is solved using PuLP's choice of Solver
solver = pulp.getSolver("GLPK_CMD")
prob.solve(solver=solver)

# The status of the solution is printed to the screen
print("Status:", pulp.LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Total Cost of Ingredients per can = ", pulp.value(prob.objective))
