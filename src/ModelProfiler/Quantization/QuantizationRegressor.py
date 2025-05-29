import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from CommonIds.NodeId import NodeId
from CommonProfile.ModelProfile import Regressor


def build_regressor(
    dataframe: pd.DataFrame,
    train_set_size: int,
    test_set_size: int,
    max_regression_degree: int,
) -> Regressor:

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

        regressors_info.append((model, train_score, test_score, degree))

    ## Sorting per max Test Score
    ## We can also add a threshold to consider when not to use the regression
    print(regressors_info)
    for _, elem in enumerate(regressors_info):
        print(f"Degree {elem[3]} ; TR Score {elem[1]} ; TE Score {elem[2]}")

    regressors_info.sort(key=lambda x: x[2], reverse=True)

    best_regressor_info = regressors_info[0]

    print("Best Regressor >> ", best_regressor_info)

    best_regressor = Regressor()
    best_regressor.set_degree(best_regressor_info[3])
    best_regressor.set_scores(best_regressor_info[1], best_regressor_info[2])

    linear_regression = best_regressor_info[0].named_steps["lin_reg"]
    poly_regression = best_regressor_info[0].named_steps["poly_features"]
    best_regressor.set_intercept(linear_regression.intercept_)

    feature_names = poly_regression.get_feature_names_out(
        input_features=X_train.columns
    )
    feature_tuples = [tuple(name.split(" ")) for name in feature_names]

    for interaction_tuple, interaction_value in zip(
        feature_tuples,
        linear_regression.coef_,
    ):
        interaction_key = tuple([NodeId(node_name) for node_name in interaction_tuple])
        best_regressor.put_interaction(
            interaction_key,
            interaction_value,
        )

    return best_regressor

    pass
