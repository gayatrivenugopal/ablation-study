import json
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

from sklearn.base import clone

from metrics import Metrics

class Ablation:
    def __init__(self, X_train, y_train, X_test, y_test, model):
        self.X_train, self.y_train, self.X_test, self.y_test, self.model = X_train, y_train, X_test, y_test, model
        self.features = self.X_train.columns.tolist()
        self.labels = self.y_train.unique()

    def run_test(self):
        """
            Run ablation test.
            Return:
                metrics (dict): key corresponds to the feature that was removed and the value corresponds to the metrics
        """
        metrics = dict()
        for feature in self.features:
            model = clone(self.model)
            train_data = self.X_train.copy()
            test_data = self.X_test.copy()
            del train_data[feature]
            del test_data[feature]
            model.fit(train_data, self.y_train)

            #evaluation
            metrics[feature] = Metrics.get_metrics(model, train_data, self.y_train, test_data, self.y_test, self.labels)
        self.model.fit(self.X_train, self.y_train)
        metrics['none'] = Metrics.get_metrics(self.model, self.X_train, self.y_train, self.X_test, self.y_test, self.labels)
        return metrics
