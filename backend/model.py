import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


class CatBoostModel:
    def __init__(self, ):
        self.cbc =  CatBoostRegressor()
        self.iterations = None
        self.depth = None

        self.X_train = None
        self.y_train = None

        self.predictions = None
        self.WEIGHTS_PATH = "/container/backend/weights"

    def process_data(self, df:pd.DataFrame, target_column_name="purchase_count_prod"):
        self.X_train = df.drop([target_column_name], axis=1)
        self.y_train = df[target_column_name]

        return self.X_train, self.y_train

    def train(self, X_train, y_train, iterations=100, depth=3):
        self.iterations = iterations
        self.depth = depth

        pooled_train = Pool(data=X_train,
                            label=y_train
                            )

        self.cbc = CatBoostRegressor(iterations=self.iterations,
                                     depth=self.depth,
                                     random_seed=42,
                                     #  task_type="GPU",
                                     #  devices="0:1",
                                     loss_function='RMSE',
                                     # eval_metric="MAE"
                                     )

        print("Обучение...")
        self.cbc.fit(pooled_train,
                     # eval_set=pooled_test,
                     use_best_model=True,
                     verbose=True)

    def save(self, weights_name: str="default"):
        self.cbc.save_model(os.path.join(self.WEIGHTS_PATH, f"{weights_name}.cbm"),
                            format="cbm",
                            export_parameters=None,
                            pool=None)

    def predict(self, X: pd.DataFrame):

        pooled = Pool(data=X)

        self.predictions = self.cbc.predict(pooled).astype(np.int32)

        return self.predictions

    def load(self, weights_name: str="default"):
        self.cbc.load_model(os.path.join(self.WEIGHTS_PATH, f"{weights_name}.cbm"), format="cbm")