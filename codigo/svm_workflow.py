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
    ini_treino = datetime.datetime.now()
    x_train, y_train_svm, y_train_class = train

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
    logging.info("ini kfold_cross_validation")
    ini_ = datetime.datetime.now()

    moc_svms_full = []
    k = dt_obj.kfolds
    list_acuracia_kfolds = []
    bit = 0
    for svm_dataset in dt_obj.list_datasets_moc_ecoc:
        # cada iteracao corresponde a um bit do moc/ecoc

        list_folds = np.arange(k)
        result_test = []
        result_train = []

        for fold in range(k):
            folds_train = np.delete(list_folds, fold)
            folds_test = np.delete(list_folds, folds_train)

            x_train, y_train_svm, y_train_class = dt_obj.get_moc_ecoc_x_y_fold(dt_obj.dict_artists_kfold,
                                                                               svm_dataset[0],
                                                                               svm_dataset[1],
                                                                               folds_train)

            x_test, y_test_svm, y_test_class = dt_obj.get_moc_ecoc_x_y_fold(dt_obj.dict_artists_kfold,
                                                                            svm_dataset[0],
                                                                            svm_dataset[1],
                                                                            folds_test)

            svm_pesos, acuracia_treino, acuracia_test = treino_svm((x_train, y_train_svm, y_train_class),
                                                                   (x_test, y_test_svm, y_test_class), config=config)
            result_train.append(acuracia_treino)
            result_test.append(acuracia_test)

        bit += 1
        list_acuracia_kfolds.append(float(cp.array(result_test).mean()))
        logging.info("acuracia media kfold treino: {} ".format(cp.array(result_train).mean()))
        logging.info("acuracia media kfold teste: {} ".format(cp.array(result_test).mean()))

    fim_ = datetime.datetime.now()
    logging.info("tempo kfold_cross_validation: {}".format(fim_ - ini_))
    logging.info("fim kfold_cross_validation")
    logging.info("kfolds lista acuracia: {}".format(list_acuracia_kfolds))


def treino_hog():
    dt_hog = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/hog_11_15_20_56",
                                 max_artists=32)

    dt_hog.load_hdf5()
    dt_hog.get_dictionary_artists()
    dt_hog.get_dictionary_kfold_test()
    dt_hog.get_dictionary_moc_ecoc_artists(ecoc=1)

    config = {}
    config['kkttol'] = 5e-2
    config['chunksize'] = 8000
    config['bias'] = []
    config['sv'] = []
    config['svcoeff'] = []
    config['normalw'] = []
    config['C'] = 10
    config['h'] = 0
    config['debug'] = True
    config['alphatol'] = 1e-2
    config['SVThresh'] = 0
    config['qpsize'] = 256
    config['logs'] = []
    config['configs'] = {}
    config['kernelpar'] = 1
    config['randomWS'] = True

    # treino_svm_total(dt_hog, config)
    # kfold_cross_validation(dt_hog, config)


def treino_lbp():
    dt_lbp = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/lbp_grid_total",
                               max_artists=128, lbp=True)

    dt_lbp.load_hdf5()
    dt_lbp.get_dictionary_artists()
    dt_lbp.get_dictionary_kfold_test()
    dt_lbp.get_dictionary_moc_ecoc_artists(ecoc=1)

    config = {}
    config['kkttol'] = 5e-2
    config['chunksize'] = 8000
    config['bias'] = []
    config['sv'] = []
    config['svcoeff'] = []
    config['normalw'] = []
    config['C'] = 10
    config['h'] = 0
    config['debug'] = True
    config['alphatol'] = 1e-2
    config['SVThresh'] = 0
    config['qpsize'] = 256
    config['logs'] = []
    config['configs'] = {}
    config['kernelpar'] = 1
    config['randomWS'] = True

    # treino_svm_total(dt_lbp, config)
    kfold_cross_validation(dt_lbp, config)

treino_lbp()

print()

