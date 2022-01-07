import sys
import logging
import h5py
import cupy as cp
import numpy as np

class dataprep:
    def __init__(self, hdf5_path, max_artists=-1, random_state=2):
        '''
        Inicia a classe
        :param hdf5_path: path com o arquivo hdf5 de origem
        :param random_state: seed para randomizar
        '''
        self.hdf5_path = hdf5_path
        self.random_state = random_state
        self.max_artists = max_artists

    def get_list_unique_artists(self):
        '''
        :return: numpy array com os artistas unicos
        '''
        f = h5py.File(path_dataset_hdf5, 'r')

        self.np_data = np.array(f['descriptor'])
        self.cp_data = cp.array(f['descriptor'])

        unique_artists = np.unique(data_[:, 0]).copy()
        np.random.seed(self.random_state)
        np.random.shuffle(unique_artists)
        if self.max_artists > -1:
            unique_artists = unique_artists[:self.max_artists]

        self.unique_artists = unique_artists
        return self.unique_artists

    def get_dictionary_artists(self, list_unique_artists=[]):
        self.dict_artists = {}
        if not list_unique_artists:
            list_unique_artists = self.get_list_unique_artists()

        for artist in list_unique_artists:
            # filtra os registros/fotografias de cada artista
            dataset = self.cp_data[cp.where(self.cp_data[:, 0] == artist)]
            # adiciona um tupla (x,y) por artista
            self.dict_artists[artist] = (dataset[:,1:], dataset[:,:1])
        return self.dict_artists







