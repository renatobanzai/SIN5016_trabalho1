from sin5016 import mlp_gpu
from sin5016 import dataprep
import sys
import logging
logging.basicConfig(filename="mlp_workflow.log", format='%(asctime)s %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import numpy as np
import cupy as cp
import datetime
import pickle

def treino_mlp(train, test, valida_teste=True, config=None):
    '''
    Treina um modelo e testa
    :param train: tupla com valores de treino(x), treino(y) (binario) e treino(yreal) classe real
    :param test: tupla com valores de treino(x), treino(y) (binario) e treino(yreal) classe real
    :param valida_teste: quando verdadeiro, printa as métricas do teste
    :param config: dicionario com a configuracao dos hiperparametros do treino do modelo
    :return:
    '''
    ini_treino = datetime.datetime.now()
    x_train, y_train_one_hot, y_train_class = train

    logging.info("X treino:{}".format(train[0].shape))
    logging.info("X teste:{}".format(test[0].shape))

    mlp_ = mlp_gpu.mlp_gpu(config=config)
    mlp_.debug = False
    mlp_.fit(x_train, y_train_one_hot)

    y_softmax, y_argmax = mlp_.predict(x_train)

    acuracia_treino = mlp_.acuracia(y_softmax, y_train_one_hot)
    logging.info("Treino Acuracia:{}".format(acuracia_treino))

    if valida_teste:
        x_test, y_test_one_hot, y_test_class = test
        y_softmax, y_argmax = mlp_.predict(x_test)
        acuracia_teste = mlp_.acuracia(y_softmax, y_test_one_hot)
        logging.info("Teste Acuracia:{}".format(acuracia_teste))

    fim_treino = datetime.datetime.now()

    logging.info("tempo treinamento {}".format(fim_treino - ini_treino))
    if valida_teste:
        return mlp_, acuracia_treino, acuracia_teste
    else:
        return mlp_, acuracia_treino

def treino_mlp_total(dt_obj, config):
    '''
    Treina e valida modelos encodados para MOC/ECOC
    :param dt_obj: objeto do tipo dataprep preenchido com os dados
    :param config: dicionario de configuracao dos hiperparametros de treinamento dos modelos
    :return:
    '''
    logging.info("ini treino_mlp_total")
    ini_full = datetime.datetime.now()
    x_train, y_train_one_hot, y_train_class = dt_obj.get_mlp_prep(dt_obj.dict_artists_train)
    x_test, y_test_one_hot, y_test_class = dt_obj.get_mlp_prep(dt_obj.dict_artists_test)

    mlp_treinado, acuracia_treino, acuracia_teste = treino_mlp((x_train, y_train_one_hot, y_train_class),
                                                               (x_test, y_test_one_hot, y_test_class),
                                                               config=config)

    pickle.dump(mlp_treinado, open("5016_mlp_gpu_{}.dat".format(len(dt_obj.unique_artists)), "wb"))
    fim_full = datetime.datetime.now()
    logging.info("tempo treino_mlp_total: {}".format(fim_full - ini_full))
    logging.info("fim treino_mlp_total")

def kfold_cross_validation(dt_obj, config):
    '''
    Performa o kfold cross validation de um modelo
    :param dt_obj: objeto dataprep preenchido com os dados a serem utilizados no treino
    :param config: dicionario com as configuracoes de hiperparametro de treino do modelo
    :return:
    '''
    logging.info("ini kfold_cross_validation")
    ini_ = datetime.datetime.now()

    moc_svms_full = []
    k = dt_obj.kfolds
    list_acuracia_kfolds = []
    bit = 0
    # percorre a lista com o dataset de cada bit do MOC/ECOC

    result_train = []
    result_test = []
    k = dt_obj.kfolds
    list_folds = np.arange(k)
    # percorre os folds
    for fold in range(k):
        folds_train = np.delete(list_folds, fold)
        folds_test = np.delete(list_folds, folds_train)

        # baseado no fold, define qual sera o cjto de treino
        x_train, y_train_one_hot, y_train_class = dt_obj.get_mlp_prep_fold(dt_obj.dict_artists_kfold,
                                                                           folds_train)
        # baseado no fold, define qual sera o cjto de teste
        x_test, y_test_one_hot, y_test_class = dt_obj.get_mlp_prep_fold(dt_obj.dict_artists_kfold,
                                                                        folds_test)

        # treina o modelo
        mlp_treinado, acuracia_treino, acuracia_teste = treino_mlp((x_train, y_train_one_hot, y_train_class),
                                                                   (x_test, y_test_one_hot, y_test_class),
                                                                   config=config)

        # guarda as metricas
        result_train.append(acuracia_treino)
        result_test.append(acuracia_teste)

        logging.info("acuracia media kfold treino: {} ".format(cp.array(result_train).mean()))
        logging.info("acuracia media kfold teste: {} ".format(cp.array(result_test).mean()))
        list_acuracia_kfolds.append(float(cp.array(result_test).mean()))

    fim_ = datetime.datetime.now()
    # loga/exibe as metricas obtidas.
    logging.info("tempo kfold_cross_validation: {}".format(fim_ - ini_))
    logging.info("fim kfold_cross_validation")
    logging.info("kfolds lista acuracia: {}".format(list_acuracia_kfolds))
    logging.info("kfolds media dos bits: {}".format(np.array(list_acuracia_kfolds).mean()))

def treino_hog():
    '''
    Treina um modelo com dados do descritor hog
    :return:
    '''
    # lendo os dados pre-processados
    max_artists = 32
    dt_hog = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/hog_11_15_20_56",
                                 max_artists=max_artists)

    # sequencia de processamentos para obter o objeto tipo dataprep com os dados da maneira necessaria ao treino
    dt_hog.load_hdf5()
    dt_hog.get_dictionary_artists()
    dt_hog.get_dictionary_kfold_test()
    dt_hog.get_dictionary_moc_ecoc_artists(ecoc=1)

    # dicionario de configuracao dos hiperparametros do modelo.
    config = {}
    config['learning_rate'] = 0.15
    config['input_layer_size'] = 576
    config['hidden_layer_size'] = 60
    config['n_iterations'] = 4000
    config['output_layer_size'] = max_artists
    config['l2'] = 0.08
    config['min_cost'] = 0.0


    # treino_mlp_total(dt_hog, config)
    kfold_cross_validation(dt_hog, config)


# def treino_lbp():
#     '''
#     Treina um SVM com uso dos dados de descritores LBP.
#     :return:
#     '''
#     dt_lbp = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/lbp_grid_total",
#                                max_artists=2545, lbp=True)
#
#     dt_lbp.load_hdf5()
#     dt_lbp.get_dictionary_artists()
#     dt_lbp.get_dictionary_kfold_test()
#     dt_lbp.get_dictionary_moc_ecoc_artists(ecoc=1)
#
#     config = {}
#     config['kkttol'] = 1e-2
#     config['chunksize'] = 4000
#     config['bias'] = []
#     config['sv'] = []
#     config['svcoeff'] = []
#     config['normalw'] = []
#     config['C'] = 1
#     config['h'] = 0
#     config['debug'] = True
#     config['alphatol'] = 0.01
#     config['SVThresh'] = 0
#     config['qpsize'] = 256
#     config['logs'] = []
#     config['configs'] = {}
#     config['kernelpar'] = 0.15
#     config['randomWS'] = True
#
#     treino_svm_total(dt_lbp, config)
#     # kfold_cross_validation(dt_lbp, config)

treino_hog()