# ML example project on diabetes data
# We'll be experimenting on the performance using different metrics and hyperparameters used


import mlflow
# from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
# Random Regressor Model is an extended optimized version of Decision Trees
#
# load & split data for training
db_data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    db_data.data, db_data.target)

desc = "This is a run description"

# connect mlflow run to mlflow tracking server by mentioning the port number
mlflow.set_tracking_uri("http://localhost:5000")

# execute the mlflow run
mlflow.set_experiment("mlflow_tracking_examples")
# with mlflow.start_run(run_name="run_without_artifacts_logged", description=desc) as run:
# create model instance by setting hyperparameters
# n_estimators -> defines the number of trees to be used in the RandomForestRegressor model
# max_depth -> determines max. number of splits each tree can take - its value may cause underfitting or overfitting
# max_features -> sets max number of features Random Forest is allowed to try in individual tree

params = {'n_estimators': 100, 'max_depth': 6, 'max_features': 3}

# automatically logs by using 17 parameters, 5 metrics & 2 tags used by scikit-learn while training and evaluating models
mlflow.autolog(log_input_examples=True, log_model_signatures=True)

# Non-nested option - running default hyperparameters

# with mlflow.start_run(run_name="autolog_with_multi_param_run") as run:
#     rfr = RandomForestRegressor(**params)
#     # train model
#     rfr.fit(X_train, y_train)


# Nested runs - running combinations of hyperparameters
# with mlflow.start_run(run_name="main_run_for_nested") as run:
#     for estimators in range(20, 100, 20):
#         # specify nested=True to indicate this is an inner run
#         with mlflow.start_run(run_name=f"nested_{estimators}_estimators", nested=True) as nested:
#             rfr = RandomForestRegressor(
#                 n_estimators=estimators, max_depth=6, max_features=3)
#             rfr.fit(X_train, y_train)


# Nested runs - using hyperparameter tuning tools (e.g. Grid Search, Random Seach, Hyperopt)
params = {
    "n_estimators": [33, 66, 200],
    "max_depth": [2, 4, 6],
    "max_features": [3, 4, 5]
}

rfr = RandomForestRegressor()
best_searcher = GridSearchCV(estimator=rfr, param_grid=params)


with mlflow.start_run(run_name="main_run_for_nested") as run:
    for estimators in range(20, 100, 20):
        # specify nested=True to indicate this is an inner run
        with mlflow.start_run(run_name=f"nested_{estimators}_estimators", nested=True) as nested:
            rfr = RandomForestRegressor(
                n_estimators=estimators, max_depth=6, max_features=3)
            rfr.fit(X_train, y_train)


# signature tells what a model gets as input and generates as output
# signature = infer_signature(X_train, rfr.predict(X_test))

# # log paramters
# mlflow.log_params(params)           # log paramters set above
# # add a single param separately
# mlflow.log_param("my_param", "my_param_value")

# # log self added metric, none defined above
# mlflow.log_metric("my_metric", 0.8)

# # set self a tag to the run
# mlflow.set_tag("my_tag", "my_tag_value")

# # log the model as an artifact (.pkl) along with its dependencies, i.e. conda.yaml, python_env.yaml, MLmodel & requirements.txt
# mlflow.sklearn.log_model(
#     sk_model=rfr,
#     artifact_path="random_forest_regressor",
#     signature=signature    # describes model input and output Schema

# )
