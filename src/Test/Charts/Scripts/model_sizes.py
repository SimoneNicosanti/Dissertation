import matplotlib.pyplot as plt
import pandas as pd


def main():

    dataframe = pd.read_csv("../../Results/model_sizes.csv")
    dataframe = dataframe.sort_values(by="num_nodes", ascending=True, inplace=False)

    # Imposta figura in formato 16:9
    plt.figure(figsize=(10, 5))

    # Plot
    x_points = dataframe["model_case"]
    plt.scatter(x_points, dataframe["num_nodes"], label="Number of Nodes")
    plt.scatter(x_points, dataframe["num_edges"], label="Number of Edges")
    plt.xticks(rotation=45)

    # Etichette e legenda
    plt.xlabel("Model Case")
    plt.ylabel("Number of Nodes/Edges")
    plt.legend()

    # Titolo
    plt.title("Model Sizes")

    # Ottimizza layout e salva
    plt.tight_layout()
    plt.grid()
    plt.savefig("../Images/model_sizes_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
