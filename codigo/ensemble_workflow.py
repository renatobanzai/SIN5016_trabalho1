from sin5016 import mlp_gpu
from sin5016 import dataprep
from sin5016 import ensemble
import sys
import logging
logging.basicConfig(filename="ensemble_workflow.log", format='%(asctime)s %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import numpy as np
import cupy as cp
import datetime
import pickle



def treino_hog_ensemble(lista_modelos):
    '''
    treina os modelos sem folds
    :param lista_modelos: lista com path dos modelos treinados
    :return:
    '''


    max_artists = 64
    dt_hog = dataprep.dataprep(hdf5_path="/home/madeleine/Documents/mestrado/5016/trabalho/data/hog_11_15_20_56",
                               max_artists=max_artists)

    # sequencia de processamentos para obter o objeto tipo dataprep com os dados da maneira necessaria ao treino
    dt_hog.load_hdf5()
    dt_hog.get_dictionary_artists()
    dt_hog.get_dictionary_kfold_test()
    dt_hog.get_dictionary_moc_ecoc_artists(ecoc=4)

    x_test, y_test_one_hot, y_test_class = dt_hog.get_mlp_prep(dt_hog.dict_artists_test)
    ens = ensemble.ensemble(dt_hog)

    for modelo in lista_modelos:
        ens.add_predictor(modelo[0], modelo[1])

    y_predito = ens.predict(x_test)

    print(ens.acuracia(y_predito, y_test_class))


lista_modelos = []
lista_modelos.append(["/home/madeleine/Documents/mestrado/5016/trabalho/codigo/svm_hog_ecoc_4_gpu_64.dat", "svm"])
lista_modelos.append(["/home/madeleine/Documents/mestrado/5016/trabalho/codigo/5016_mlp_gpu_64_relu.dat", "mlp"])
lista_modelos.append(["/home/madeleine/Documents/mestrado/5016/trabalho/codigo/5016_mlp_gpu_64_sigmoid.dat", "mlp"])
lista_modelos.append(["/home/madeleine/Documents/mestrado/5016/trabalho/codigo/5016_mlp_gpu_64.dat", "mlp"])
treino_hog_ensemble(lista_modelos)