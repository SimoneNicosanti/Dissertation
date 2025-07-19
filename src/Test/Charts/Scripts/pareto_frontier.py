import pandas as pd
from matplotlib import pyplot as plt


def main():

    df = pd.read_csv("../../Results/ParetoFrontier/pareto_frontier.csv")
    latency_cost = df["latency_cost"]
    energy_cost = df["energy_cost"]

    plt.scatter(latency_cost, energy_cost)
    plt.show()
    pass


if __name__ == "__main__":
    main()
