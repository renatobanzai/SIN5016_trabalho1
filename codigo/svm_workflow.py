from sin5016 import svm_gpu
from sin5016 import dataprep
import sys
import logging
logging.basicConfig(filename="svm_workflow.py.log", format='%(asctime)s %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import numpy as np
import cupy as cp
import datetime
import pickle

def treino_svm(train, test, valida_teste=True, config=None):
    '''
    Treina um modelo e testa
    :param train: tupla com valores de treino(x), treino(y) (binario) e treino(yreal) classe real
    :param test: tupla com valores de treino(x), treino(y) (binario) e treino(yreal) classe real
    :param valida_teste: quando verdadeiro, printa as métricas do teste
    :param config: dicionario com a configuracao dos hiperparametros do treino do modelo
    :return:
    '''
    ini_treino = datetime.datetime.now()
    x_train, y_train_svm, y_train_class = train

    logging.info("X treino:{}".format(train[0].shape))
    logging.info("X teste:{}".format(test[0].shape))

    svm_ = svm_gpu.svm_gpu(x_train, y_train_svm, config=config)
    svm_.debug = False
    svm_.prep()
    svm_.fit()

    y_predict_train, y1_predict_train = svm_.predict(x_train)
    acuracia_treino = svm_gpu.acuracia(y_train_svm, y_predict_train)

    logging.info("Treino Acuracia:{}".format(acuracia_treino))

    if valida_teste:
        x_test, y_test_svm, y_test_class = test
        y_predict_test, y1_predict_test = svm_.predict(x_test)
        acuracia_test = svm_gpu.acuracia(y_test_svm, y_predict_test)
        logging.info("Teste Acuracia:{}".format(acuracia_test))

    fim_treino = datetime.datetime.now()

    logging.info("tempo treinamento {}".format(fim_treino - ini_treino))
    if valida_teste:
        return svm_.return_instance_for_predict(), acuracia_treino, acuracia_test
    else:
        return svm_.return_instance_for_predict(), acuracia_treino

def treino_svm_total(dt_obj, config):
    '''
    Treina e valida modelos encodados para MOC/ECOC
    :param dt_obj: objeto do tipo dataprep preenchido com os dados
    :param config: dicionario de configuracao dos hiperparametros de treinamento dos modelos
    :return:
    '''
    logging.info("ini treino_svm_total")
    ini_full = datetime.datetime.now()

    moc_svms_full = []
    for svm_dataset in dt_obj.list_datasets_moc_ecoc:
        x_train, y_train_svm, y_train_class = dt_obj.get_moc_ecoc_x_y(dt_obj.dict_artists_train, svm_dataset[0],
                                                                      svm_dataset[1])
        x_test, y_test_svm, y_test_class = dt_obj.get_moc_ecoc_x_y(dt_obj.dict_artists_test, svm_dataset[0],
                                                                   svm_dataset[1])

        svm_pesos, a, b = treino_svm((x_train, y_train_svm, y_train_class), (x_test, y_test_svm, y_test_class),
                                     config=config)
        moc_svms_full.append(svm_pesos)

    pickle.dump(moc_svms_full, open("5016_svm_hog_moc_gpu_{}.dat".format(len(dt_obj.unique_artists)), "wb"))

    resultados = []
    for svm_treinado in moc_svms_full:
        Ysvm_train, Y1svm_train = svm_treinado.predict(x_train)
        Ysvm_train[Ysvm_train < 0] = '0'
        Ysvm_train[Ysvm_train > 0] = '1'
        resultados.append(np.array(Ysvm_train.get()).astype(int).astype(str))

    dt_obj.decoder_resultados_oc(dt_obj.dict_moc_ecoc_artists, resultados, y_train_class, "treino")

    resultados_teste = []
    for svm_treinado in moc_svms_full:
        Ysvm_train, Y1svm_train = svm_treinado.predict(x_test)
        Ysvm_train[Ysvm_train < 0] = '0'
        Ysvm_train[Ysvm_train > 0] = '1'
        resultados_teste.append(np.array(Ysvm_train.get()).astype(int).astype(str))

    dt_obj.decoder_resultados_oc(dt_obj.dict_moc_ecoc_artists, resultados_teste, y_test_class, "teste")

    fim_full = datetime.datetime.now()
    logging.info("tempo treino_svm_total: {}".format(fim_full - ini_full))
    logging.info("fim treino_svm_total")

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
    for svm_dataset in dt_obj.list_datasets_moc_ecoc:
        list_folds = np.arange(k)
        result_test = []
        result_train = []

        # percorre os folds
        for fold in range(k):
            folds_train = np.delete(list_folds, fold)
            folds_test = np.delete(list_folds, folds_train)

            # baseado no fold, define qual sera o cjto de treino
            x_train, y_train_svm, y_train_class = dt_obj.get_moc_ecoc_x_y_fold(dt_obj.dict_artists_kfold,
                                                                               svm_dataset[0],
                                                                               svm_dataset[1],
                                                                               folds_train)
            # baseado no fold, define qual sera o cjto de teste
            x_test, y_test_svm, y_test_class = dt_obj.get_moc_ecoc_x_y_fold(dt_obj.dict_artists_kfold,
                                                                            svm_dataset[0],
                                                                            svm_dataset[1],
                                                                            folds_test)

            # treina o modelo
            svm_pesos, acuracia_treino, acuracia_test = treino_svm((x_train, y_train_svm, y_train_class),
                                                                   (x_test, y_test_svm, y_test_class), config=config)

            # guarda as metricas
            result_train.append(acuracia_treino)
            result_test.append(acuracia_test)

        bit += 1
        # executados os folds, acumula a media da acuracia daquele "bit"
        list_acuracia_kfolds.append(float(cp.array(result_test).mean()))
        logging.info("acuracia media kfold treino: {} ".format(cp.array(result_train).mean()))
        logging.info("acuracia media kfold teste: {} ".format(cp.array(result_test).mean()))

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
    dt_hog = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/hog_11_15_20_56",
                                 max_artists=2544)

    # sequencia de processamentos para obter o objeto tipo dataprep com os dados da maneira necessaria ao treino
    dt_hog.load_hdf5()
    dt_hog.get_dictionary_artists()
    dt_hog.get_dictionary_kfold_test()
    dt_hog.get_dictionary_moc_ecoc_artists(ecoc=1)

    # dicionario de configuracao dos hiperparametros do modelo.
    config = {}
    config['kkttol'] = 0.05
    config['chunksize'] = 4000
    config['bias'] = []
    config['sv'] = []
    config['svcoeff'] = []
    config['normalw'] = []
    config['C'] = 0.1
    config['h'] = 0
    config['debug'] = True
    config['alphatol'] = 5e-2
    config['SVThresh'] = 0
    config['qpsize'] = 256
    config['logs'] = []
    config['configs'] = {}
    config['kernelpar'] = 0.95
    config['randomWS'] = True

    treino_svm_total(dt_hog, config)
    # kfold_cross_validation(dt_hog, config)


def treino_lbp():
    '''
    Treina um SVM com uso dos dados de descritores LBP.
    :return:
    '''
    dt_lbp = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/lbp_grid_total",
                               max_artists=2545, lbp=True)

    dt_lbp.load_hdf5()
    dt_lbp.get_dictionary_artists()
    dt_lbp.get_dictionary_kfold_test()
    dt_lbp.get_dictionary_moc_ecoc_artists(ecoc=1)

    config = {}
    config['kkttol'] = 1e-2
    config['chunksize'] = 4000
    config['bias'] = []
    config['sv'] = []
    config['svcoeff'] = []
    config['normalw'] = []
    config['C'] = 1
    config['h'] = 0
    config['debug'] = True
    config['alphatol'] = 0.01
    config['SVThresh'] = 0
    config['qpsize'] = 256
    config['logs'] = []
    config['configs'] = {}
    config['kernelpar'] = 0.15
    config['randomWS'] = True

    treino_svm_total(dt_lbp, config)
    # kfold_cross_validation(dt_lbp, config)

treino_lbp()