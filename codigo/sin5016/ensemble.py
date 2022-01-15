from .svm_gpu import svm_gpu
from .mlp_gpu import mlp_gpu
from .dataprep import dataprep
import cupy as cp
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

class ecoc:
    def __init__(self, dict_modelo_ecoc):
        self.svms = dict_modelo_ecoc["svms"]
        self.dt_obj = dataprep("")
        self.dt_obj.dict_moc_ecoc_artists = dict_modelo_ecoc["dict_moc"]
        self.dt_obj.dict_matriz_binarios = dict_modelo_ecoc["dict_matriz_binarios"]

    def predict(self, X):
        resultados_teste = []
        for svm_treinado in self.svms:
            Ysvm_train, Y1svm_train = svm_treinado.predict(X)
            Ysvm_train[Ysvm_train < 0] = '0'
            Ysvm_train[Ysvm_train > 0] = '1'
            resultados_teste.append(np.array(Ysvm_train.get()).astype(int).astype(str))

        return self.dt_obj.decoder_resultados_oc(self.dt_obj.dict_moc_ecoc_artists, resultados_teste, None, "teste", calcular_acuracia=False)

class ensemble:
    def __init__(self, dt_obj):
        self.predictors = []

    def add_predictor(self, predictor_path, predictor_type="mlp"):
        with open(predictor_path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            if predictor_type == "mlp":
                predictor = pickle.load(f)
            elif predictor_type=="svm":
                predictor = ecoc(pickle.load(f))
        self.predictors.append([predictor, predictor_type])

    def predict(self, X):
        results = []
        for predictor in self.predictors:
            if predictor[1] =="svm":
                results.append(predictor[0].predict(X))
            elif predictor[1] == "mlp":
                one_hot_encoder = predictor[0].one_hot_encoder
                res = list(np.array(predictor[0].predict(X)[1].get()))
                results.append(cp.array([one_hot_encoder.categories_[0][x] for x in res]).reshape(-1,1))

        ensemble_result = []
        for line in range(results[0].shape[0]):
            line_result = []
            for column in range(len(results)):
                line_result.append(results[column][line])
            ensemble_result.append(cp.array(line_result))

        final_result = []
        for e_result in ensemble_result:
            vals, counts = cp.unique(e_result[:, 0], return_counts=True)
            max_count = np.max(counts)
            if max_count > 1:
                res = vals[counts == max_count][0]
            else:
                res = e_result[0]

            final_result.append(res)

        return cp.array([float(x) for x in final_result]).reshape(-1,1)

    def acuracia(self, y_predito, y_desejado):
        return (y_predito == y_desejado).astype(int).mean()





