from .svm_gpu import svm_gpu
from .mlp_gpu import mlp_gpu
from .dataprep import dataprep
import cupy as cp
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

class ecoc:
    '''
    Criamos essa classe para representar um conjunto de SVMs
    '''
    def __init__(self, dict_modelo_ecoc):
        '''
        Instancia a classe
        :param dict_modelo_ecoc: dicionario contendo os svms e seus atributos
        '''
        self.svms = dict_modelo_ecoc["svms"]
        self.dt_obj = dataprep("")
        self.dt_obj.dict_moc_ecoc_artists = dict_modelo_ecoc["dict_moc"]
        self.dt_obj.dict_matriz_binarios = dict_modelo_ecoc["dict_matriz_binarios"]

    def predict(self, X):
        '''
        faz o predict
        :param X: features
        :return: y das classes preditas
        '''
        resultados_teste = []
        for svm_treinado in self.svms:
            Ysvm_train, Y1svm_train = svm_treinado.predict(X)
            Ysvm_train[Ysvm_train < 0] = '0'
            Ysvm_train[Ysvm_train > 0] = '1'
            resultados_teste.append(np.array(Ysvm_train.get()).astype(int).astype(str))

        return self.dt_obj.decoder_resultados_oc(self.dt_obj.dict_moc_ecoc_artists, resultados_teste, None, "teste", calcular_acuracia=False)

class ensemble:
    '''

    '''
    def __init__(self, dt_obj):
        self.predictors = []

    def add_predictor(self, predictor_path, predictor_type="mlp"):
        '''
        adiciona um preditor baseado no endereco de seu modelo serializado
        :param predictor_path: local do pickle do preditor
        :param predictor_type: svm, mlp
        :return:
        '''
        with open(predictor_path, 'rb') as f:
            if predictor_type == "mlp":
                # ao ser mlp, basta adicionar
                predictor = pickle.load(f)
            elif predictor_type=="svm":
                # caso seja svm, necessita usar a classe ecoc para tratar o ecoc/moc
                predictor = ecoc(pickle.load(f))
        self.predictors.append([predictor, predictor_type])

    def predict(self, X):
        '''
        combina os componentes do ensemble para obter um y
        :param X: features
        :return:
        '''
        results = []
        # percorre a lista de preditores e faz o predict de cada um
        for predictor in self.predictors:
            if predictor[1] =="svm":
                # svm ja retorna as classes
                results.append(predictor[0].predict(X))
            elif predictor[1] == "mlp":
                # mlp precisa decodificar o one_hot
                one_hot_encoder = predictor[0].one_hot_encoder
                res = list(np.array(predictor[0].predict(X)[1].get()))
                results.append(cp.array([one_hot_encoder.categories_[0][x] for x in res]).reshape(-1,1))

        ensemble_result = []
        # gera uma lista, a cada linha possui o resultado de cada preditor.
        for line in range(results[0].shape[0]):
            line_result = []
            for column in range(len(results)):
                line_result.append(results[column][line])
            ensemble_result.append(cp.array(line_result))



        final_result = []
        # calcula os votos em cada classe
        for e_result in ensemble_result:
            vals, counts = cp.unique(e_result[:, 0], return_counts=True)
            max_count = np.max(counts)
            if max_count > 1:
                # caso tenha uma classe com mais votos eh a escolhia
                # the winner takes all
                res = vals[counts == max_count][0]
            else:
                # caso empate nos votos, usa-se o melhor preditor (o primeiro)
                res = e_result[0]

            final_result.append(res)

        return cp.array([float(x) for x in final_result]).reshape(-1,1)

    def acuracia(self, y_predito, y_desejado):
        '''
        calcula a acuracia comparando 2 ys
        :param y_predito:
        :param y_desejado:
        :return:
        '''
        return (y_predito == y_desejado).astype(int).mean()





