import sys
import logging
import h5py
import cupy as cp
import numpy as np
import math
import random
import datetime
from sklearn.preprocessing import OneHotEncoder

class dataprep:
    def __init__(self, hdf5_path, max_artists=-1, random_state=2, lbp=False):
        '''
        Inicia a classe
        :param hdf5_path: path com o arquivo hdf5 de origem
        :param random_state: seed para randomizar
        '''
        self.hdf5_path = hdf5_path
        self.random_state = random_state
        self.max_artists = max_artists
        self.lbp = lbp


    def load_hdf5(self, min_photos_by_artist=7):
        '''
        :return: numpy array com os artistas unicos
        '''
        f = h5py.File(self.hdf5_path, 'r')
        self.np_data = np.array(f['descriptor'])
        self.cp_data = cp.array(f['descriptor'])

        unique_artists, photos_by_artist = np.unique(self.np_data[:, 0], return_counts=True)

        unique_artists = unique_artists[photos_by_artist >= min_photos_by_artist]

        np.random.seed(self.random_state)

        np.random.shuffle(unique_artists)
        if self.max_artists > -1:
            unique_artists = unique_artists[:self.max_artists]

        self.unique_artists = unique_artists
        return self.unique_artists

    def get_dictionary_artists(self, list_unique_artists=[]):
        self.dict_artists = {}
        if not list_unique_artists:
            list_unique_artists = self.unique_artists

        for artist in list_unique_artists:
            # filtra os registros/fotografias de cada artista
            dataset = self.cp_data[cp.where(self.cp_data[:, 0] == artist)]
            # adiciona um tupla (x,y) por artista
            if self.lbp:
                self.dict_artists[artist] = (dataset[:,1:]/255, dataset[:,:1])
            else:
                self.dict_artists[artist] = (dataset[:, 1:], dataset[:, :1])
        return self.dict_artists

    def get_dictionary_kfold_test(self, train_percent=0.8, k=5):
        self.kfolds = k
        self.dict_artists_kfold = {}
        self.dict_artists_test = {}
        self.dict_artists_train = {}
        folds_fotos = 0
        test_fotos = 0
        lost_artist = False
        for artist in self.dict_artists.keys():

            artist_photos = self.dict_artists[artist][0].shape[0]
            split_train = math.floor(artist_photos*train_percent)
            if split_train < 1 or artist_photos < 2:
                lost_artist = True
                logging.warning("artista {} nao possui registros suficientes para treino e teste.".format(artist))
            else:
                if split_train < k:
                    lost_artist = True
                    logging.warning("artista {} nao possui registros suficientes kfold.".format(artist))
                else:
                    train_x = self.dict_artists[artist][0][:split_train]
                    train_y = self.dict_artists[artist][1][:split_train]

                    folds_x = cp.array_split(train_x, k)
                    folds_y = cp.array_split(train_y, k)

                    self.dict_artists_kfold[artist] = (folds_x, folds_y)

                    folds_fotos += split_train
                    test_fotos += (artist_photos - split_train)

                    test_x = self.dict_artists[artist][0][split_train:]
                    test_y = self.dict_artists[artist][1][split_train:]

                    self.dict_artists_train[artist] = (train_x, train_y)
                    self.dict_artists_kfold[artist] = (folds_x, folds_y)
                    self.dict_artists_test[artist] = (test_x, test_y)


        # por restringir artistas, muda a lista de artistas unicos
        if lost_artist:
            self.unique_artists = list(self.dict_artists_kfold.keys())

        logging.info("add {} artistas / {} fotos ao dict_artists_kfolds.".format(len(self.dict_artists_kfold.keys()), folds_fotos))
        logging.info("add {} artistas / {} fotos ao dict_artists_tests.".format(len(self.dict_artists_test.keys()), test_fotos))

        return self.dict_artists_train, self.dict_artists_kfold, self.dict_artists_test

    def get_dictionary_moc_ecoc_artists(self, ecoc=1, lbp=False):
        # quebra o dataset em treino ,teste e validacao

        if ecoc <= 1:
            logging.info("ini treino MOC")
        else:
            logging.info("ini treino ECOC com %fx o numero de bits.", ecoc)

        if lbp:
            logging.info("Descritor: LBP")
        else:
            logging.info("Descritor: HOG")

        qty_classes = len(self.unique_artists)
        logging.info("gerando matriz de binarios para : %i artistas", qty_classes)

        # gera uma matriz de binários para atribuir às classes
        self.moc_ecoc_qty = math.ceil(math.ceil(math.log2(qty_classes)) * ecoc)

        binary_format = "{0:0" + str(self.moc_ecoc_qty) + "b}"

        # lista_randomica
        max_random_val = 2 ** self.moc_ecoc_qty

        np.random.seed(self.random_state)
        if ecoc > 1:
            randomlist = []
            # evita repeticao de randomicos
            while len(set(randomlist)) < qty_classes:
                randomlist = list(set(randomlist))
                n = np.random.randint(0, max_random_val)
                randomlist.append(n)
        else:
            randomlist = [x for x in range(max_random_val)]

        matriz_binarios = []
        for i in randomlist:
            bin_rep = binary_format.format(i)
            matriz_binarios.append(bin_rep)

        np.random.shuffle(matriz_binarios)

        self.dict_moc_ecoc_artists = {}
        # PARA FACILITAR O CALCULO DE DISTANCIA
        self.dict_matriz_binarios = {}
        for i in range(qty_classes):
            self.dict_moc_ecoc_artists[self.unique_artists[i]] = matriz_binarios[i]
            # PARA FACILITAR O CALCULO DE DISTANCIA
            self.dict_matriz_binarios[matriz_binarios[i]] = self.bit_to_list(matriz_binarios[i])

        logging.info("Matriz de bits para ECOC/MOC {}".format(self.dict_moc_ecoc_artists))


        self.list_datasets_moc_ecoc = []
        for i in range(self.moc_ecoc_qty):
            classes_dataset_0 = []
            classes_dataset_1 = []
            for classe in self.dict_moc_ecoc_artists:
                if self.dict_moc_ecoc_artists[classe][i] == "1":
                    classes_dataset_1.append(classe)
                else:
                    classes_dataset_0.append(classe)
            self.list_datasets_moc_ecoc.append((classes_dataset_0, classes_dataset_1))

        return self.dict_moc_ecoc_artists, self.list_datasets_moc_ecoc

    def get_moc_ecoc_x_y(self, dict_data, lista_classe_0, lista_classe_1, lbp=False):
        for i in range(len(lista_classe_0)):
            if i == 0:
                x_train_0 = dict_data[lista_classe_0[i]][0]
                y_classes_reais_0 = dict_data[lista_classe_0[i]][1]
            else:
                x_train_0 = cp.concatenate([x_train_0, dict_data[lista_classe_0[i]][0]])
                y_classes_reais_0 = cp.concatenate(
                    [y_classes_reais_0, dict_data[lista_classe_0[i]][1]])

        for i in range(len(lista_classe_1)):
            if i==0:
                x_train_1 = dict_data[lista_classe_1[i]][0]
                y_classes_reais_1 = dict_data[lista_classe_1[i]][1]
            else:
                x_train_1 = cp.concatenate([x_train_1, dict_data[lista_classe_1[i]][0]])
                y_classes_reais_1 = cp.concatenate(
                    [y_classes_reais_1, dict_data[lista_classe_1[i]][1]])

        x_train = cp.concatenate([x_train_0, x_train_1])
        y_train_0 = cp.full(x_train_0.shape[0], -1)
        y_train_1 = cp.full(x_train_1.shape[0], 1)
        y_train = cp.concatenate([y_train_0, y_train_1])
        y_classes_reais = cp.concatenate([y_classes_reais_0, y_classes_reais_1])
        y_train = y_train.reshape(-1, 1)
        y_classes_reais = y_classes_reais.reshape(-1, 1)
        return x_train, y_train, y_classes_reais

    def get_moc_ecoc_x_y_fold(self, dict_data, lista_classe_0, lista_classe_1, folds=[], lbp=False):
        j = 0
        k = 0
        for fold in folds:
            for i in range(len(lista_classe_0)):
                if j == 0:
                    x_train_0 = dict_data[lista_classe_0[i]][0][fold]
                    y_classes_reais_0 = dict_data[lista_classe_0[i]][1][fold]
                    j = 1
                else:
                    x_train_0 = cp.concatenate([x_train_0, dict_data[lista_classe_0[i]][0][fold]])
                    y_classes_reais_0 = cp.concatenate(
                        [y_classes_reais_0, dict_data[lista_classe_0[i]][1][fold]])

            for i in range(len(lista_classe_1)):
                if k==0:
                    x_train_1 = dict_data[lista_classe_1[i]][0][fold]
                    y_classes_reais_1 = dict_data[lista_classe_1[i]][1][fold]
                    k = 1
                else:
                    x_train_1 = cp.concatenate([x_train_1, dict_data[lista_classe_1[i]][0][fold]])
                    y_classes_reais_1 = cp.concatenate(
                        [y_classes_reais_1, dict_data[lista_classe_1[i]][1][fold]])

        x_train = cp.concatenate([x_train_0, x_train_1])
        y_train_0 = cp.full(x_train_0.shape[0], -1)
        y_train_1 = cp.full(x_train_1.shape[0], 1)
        y_train = cp.concatenate([y_train_0, y_train_1])
        y_classes_reais = cp.concatenate([y_classes_reais_0, y_classes_reais_1])
        y_train = y_train.reshape(-1, 1)
        y_classes_reais = y_classes_reais.reshape(-1, 1)
        return x_train, y_train, y_classes_reais

    def hamming_distance_list(self, bit_1, bit_2):  # 0.06
        hamming = 0
        i = 0
        for x in bit_1:
            hamming += abs(bit_2[i] - x)
            i += 1
        return hamming

    def min_hamming_distance_list(self, val, list_vals, max_hamming=-1, classe_default=""):
        # ini = datetime.datetime.now()
        min_val = classe_default
        min_hamming = max_hamming
        for j in list_vals:
            hd = self.hamming_distance_list(val, j)
            if hd == 1:
                return self.list_to_bit(j)
            if hd < min_hamming:
                min_hamming = hd
                min_val = j
        return self.list_to_bit(min_val)

    def hamming_distance(self, bit_1, bit_2):
        hamming = 0
        # hamming = abs(np.array(list(bit_1)).astype(np.int8) - np.array(list(bit_2)).astype(np.int8)).sum()
        size = len(bit_1)
        for x in range(size):
            hamming += abs(int(bit_2[x]) - int(bit_1[x]))
        return hamming

    def min_hamming_distance(self, val, list_vals, max_hamming=-1):
        # ini = datetime.datetime.now()
        min_hamming = max_hamming
        if max_hamming < 0:
            min_hamming = len(val) + 1
        for j in list_vals:
            hd = self.hamming_distance(val, j)
            if hd==1:
                return j
            if hd < min_hamming:
                min_hamming = hd
                min_val = j
        # fim = datetime.datetime.now()
        # tempo = fim - ini
        return min_val

    def bit_to_list(self, bit_string):
        list_bin = list(bit_string)
        return [int(x) for x in list_bin]

    def list_to_bit(self, list_bit):
        list_str = [str(x) for x in list_bit]
        return "".join(list_str)


    def decoder_resultados_oc(self, dict_moc, resultados, y_classes_reais, tipo=""):
        logging.info("decode de {} linhas.".format(len(resultados[0])))
        ini = datetime.datetime.now()
        # cria um dicionário inverso para o decode
        dict_classes = {}
        for classe in dict_moc.keys():
            dict_classes[dict_moc[classe]] = classe
            classe_default = self.bit_to_list(dict_moc[classe])

        res_moc = []
        for res in range(len(resultados[0])):
            lista_res = "".join([res_y[res][0] for res_y in resultados])
            res_moc.append(lista_res)

        res_moc_classes = []

        # para não fazer a contagem item a item no metodo min_hamming_distante
        max_hamming = len(res_moc[0])

        for x in res_moc:
            if x in dict_classes.keys():
                res_moc_classes.append(dict_classes[x])
            else:
                res_bit = self.min_hamming_distance_list(self.bit_to_list(x),
                                                         self.dict_matriz_binarios.values(),
                                                         max_hamming=max_hamming,
                                                         classe_default=classe_default)

                res_moc_classes.append(dict_classes[res_bit])

        res_moc_classes = cp.array(res_moc_classes).reshape(-1, 1)

        res_moc_classes = res_moc_classes.reshape(-1, 1)
        y_classes_reais = y_classes_reais.reshape(-1, 1)
        fim = datetime.datetime.now()
        logging.info("decode tempo: {}".format(fim-ini))
        logging.info("ecoc: acuracia %s %f", tipo, (res_moc_classes == y_classes_reais).astype(int).mean())

    def one_hot(self, y):
        enc = OneHotEncoder(sparse=False, categories='auto')
        one_hot_y = np.array(y.get())
        one_hot_y = enc.fit_transform(one_hot_y.reshape(len(one_hot_y), -1))
        one_hot_y = cp.array(one_hot_y)


        return one_hot_y

    def get_mlp_prep(self, dict_data, lbp=False):
        lista_classes = list(dict_data.keys())
        for i in range(len(lista_classes)):
            if i == 0:
                x_train = dict_data[lista_classes[i]][0]
                y_classes_reais = dict_data[lista_classes[i]][1]
            else:
                x_train = cp.concatenate([x_train, dict_data[lista_classes[i]][0]])
                y_classes_reais = cp.concatenate(
                    [y_classes_reais, dict_data[lista_classes[i]][1]])

        y_train = self.one_hot(y_classes_reais)

        return x_train, y_train, y_classes_reais

    def get_mlp_prep_fold(self, dict_data, folds=[], lbp=False):
        j = 0
        k = 0
        lista_classes = list(dict_data.keys())
        for fold in folds:
            for i in range(len(lista_classes)):
                if i == 0:
                    x_train = dict_data[lista_classes[i]][0][fold]
                    y_classes_reais = dict_data[lista_classes[i]][1][fold]
                else:
                    x_train = cp.concatenate([x_train, dict_data[lista_classes[i]][0][fold]])
                    y_classes_reais = cp.concatenate(
                        [y_classes_reais, dict_data[lista_classes[i]][1][fold]])

        y_train = self.one_hot(y_classes_reais)

        # y_train = y_train.reshape(-1, 1)
        y_classes_reais = y_classes_reais.reshape(-1, 1)
        return x_train, y_train, y_classes_reais













