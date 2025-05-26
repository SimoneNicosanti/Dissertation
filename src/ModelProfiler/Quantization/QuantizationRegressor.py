import networkx as nx
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from CommonProfile.NodeId import NodeId


def build_regressor(
    dataframe: pd.DataFrame,
    train_set_size: int,
    test_set_size: int,
    max_regression_degree: int,
) -> tuple:

    train_set, test_set = dataframe[:train_set_size], dataframe[train_set_size:]

    X_train, Y_train = train_set.drop("noise", axis=1), train_set["noise"]
    X_test, Y_test = test_set.drop("noise", axis=1), test_set["noise"]

    regressors_info = []
    for degree in range(1, max_regression_degree + 1):
        model = Pipeline(
            [
                (
                    "poly_features",
                    PolynomialFeatures(
                        degree=degree,
                        include_bias=False,
                        interaction_only=True,
                    ),
                ),
                ("lin_reg", LinearRegression()),
            ]
        )

        model.fit(X_train, Y_train)
        train_score = model.score(X_train, Y_train)
        test_score = model.score(X_test, Y_test)

        regressors_info.append((model, train_score, test_score))

    ## Sorting per max Test Score
    ## We can also add a threshold to consider when not to use the regression
    print(regressors_info)
    for idx, elem in enumerate(regressors_info):
        print(f"Degree {idx + 1} ; TR Score {elem[1]} ; TE Score {elem[2]}")

    regressors_info.sort(key=lambda x: x[2], reverse=True)

    best_regressor_info = regressors_info[0]

    print("Best Regressor >> ", best_regressor_info)

    return best_regressor_info

    pass


def embed_regressor_in_profile(
    model_graph: nx.DiGraph,
    dataframe: pd.DataFrame,
    regressor: Pipeline,
    train_score: float,
    test_score: float,
) -> nx.DiGraph:

    # Estrai lo step PolynomialFeatures dalla pipeline
    poly = regressor.named_steps["poly_features"]

    column_names = dataframe.columns[:-1]

    # Ottieni i nomi delle feature polinomiali
    feature_names = poly.get_feature_names_out(input_features=column_names)
    feature_tuples = [tuple(name.split(" ")) for name in feature_names]

    # Ora puoi stampare i coefficienti con i nomi
    lin_reg = regressor.named_steps["lin_reg"]
    coeffs = lin_reg.coef_

    model_info = {}
    model_info["coef"] = {}
    model_info["intercept"] = {}

    # Stampa ordinata
    for name_list, coef in zip(feature_tuples, coeffs):
        node_id_tuple = tuple([NodeId(node_name) for node_name in name_list])
        model_info["coef"][node_id_tuple] = coef

    model_info["intercept"] = lin_reg.intercept_

    model_info["train_score"] = train_score
    model_info["test_score"] = test_score

    model_graph.graph["regressor"] = model_info

    print("Embedded Regressor")

    pass
