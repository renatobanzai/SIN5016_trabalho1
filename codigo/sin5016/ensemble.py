import dataprep
import mlp_gpu
import svm_gpu
import cupy as cp
import numpy as np
import pickle

class ensemble:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.predictors = []

    def add_predictor(self, predictor_path):
        predictor = pickle.load(predictor_path)
        self.predictors.append(predictor)

    def predict(self):
        results = []
        for predictor in self.predictors:
            results.append(predictor.predict(self.X))

        ensemble_result = []
        for line in range(results[0]):
            line_result = []
            for column in range(len(results)):
                line_result.append(results[line][column])
            ensemble_result.append(cp.array(line_result))

        final_result = []
        for e_result in ensemble_result:
            vals, counts = cp.unique(results[:, 0], return_counts=True)
            max_count = np.max(counts)
            if max_count > 1:
                res = vals[counts == max_count]
            else:
                res = vals[0]
        return cp.array(final_result).reshape(-1,1)

    def acuracia(self, y_predito, y_desejado):
        return (y_predito == y_desejado).astype(int).mean()





