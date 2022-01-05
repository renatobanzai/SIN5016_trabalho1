import numpy as np
import cupy as cp
import math
from cvxopt import matrix, solvers
import matplotlib.pylab as plt
import h5py
import datetime
import pickle
import random
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.kkttol = 5e-4
        self.chunksize = 4000
        self.bias = []
        self.sv = []
        self.svcoeff = []
        self.normalw = []
        self.C = 2000
        self.h = 0.01
        self.debug = True
        self.alphatol = 1e-10
        self.SVThresh = 0.
        self.qpsize = 1300
        self.logs = []
        self.configs = {}

    def prep(self):
        self.N, self.ne = self.X.shape
        self.class0 = self.Y == -1
        self.class1 = self.Y == 1

        self.Y = self.Y.reshape(self.N, 1)
        self.qpsize = min(self.N, self.qpsize)

        if self.debug: print("Working set possui {} exemplos de classe positiva e {} exemplos de classe negativa.".format(self.Y[self.class1].shape[0], self.Y[self.class0].shape[0]))


        self.C = np.full((self.N, 1), self.C)
        self.alpha = np.zeros((self.N, 1))
        self.randomWS = True



    def trainSVM(self):
        X = self.X
        Y = self.Y

        saida_svm = np.zeros((self.N,1))
        alphaOld = np.copy(self.alpha)
        if self.Y[self.class1].shape[0] == self.N:
            self.bias = 1
            return

        if self.Y[self.class0].shape[0] == self.N:
            self.bias = -1
            return

        iteracao = 0
        workset = np.full((self.N,1), False)
        sameWS = 0
        self.bias = 0
        while True:
            if self.debug: print("Iteracao{}".format(iteracao))

            # Passo 1: determina os vetores de suporte
            self.findSV()

            # Passo 2: Encontra a saída para o SVM
            if iteracao == 0:
                changedSV = self.svind
                changedAlpha = self.alpha[changedSV]
                saida_svm = np.zeros((self.N, 1))
            else:
                changedSV = np.flatnonzero(alphaOld != self.alpha)
                changedAlpha = self.alpha[changedSV] - alphaOld[changedSV]

            # Função de kernel RBF
            chunks1 = math.ceil(self.N / self.chunksize)
            chunks2 = math.ceil(len(changedSV) / self.chunksize)

            for ch1 in range(chunks1):
                ini_ind1 = (ch1)*self.chunksize
                fim_ind1 = min(self.N, (ch1+1)*self.chunksize)
                for ch2 in range(chunks2):
                    ini_ind2 = (ch2)*self.chunksize
                    fim_ind2 = min(len(changedSV), (ch2+1)*self.chunksize)
                    K12 = self.calc_rbf(X[ini_ind1:fim_ind1, :], X[changedSV][ini_ind2:fim_ind2])
                    coeff = changedAlpha[ini_ind2:fim_ind2]*Y[changedSV[ini_ind2:fim_ind2]]

            # Passo 3: Calcule o bias da função de decisão
            workSV = np.flatnonzero(np.logical_and(self.SVnonBound, workset))
            if not workSV.size == 0:
                self.bias = np.mean(Y[workSV] - saida_svm[workSV])


            # Passo 4: Calcula as condicoes de KKT
            KKT = (saida_svm+self.bias)*self.Y-1
            KKTViolations1 = np.logical_and(self.SVnonBound, (abs(KKT)>self.kkttol))
            KKTViolations2 = np.logical_and(self.SVBound, (abs(KKT) > self.kkttol))
            KKTViolations3 = np.logical_and(np.logical_not(self.SV), (abs(KKT) > self.kkttol))
            KKTViolations = np.logical_or(KKTViolations1, KKTViolations2)
            KKTViolations = np.logical_or(KKTViolations, KKTViolations3)

            count_kkt = len(np.flatnonzero(KKTViolations))

            if count_kkt == 0:
                # sem violacoes, terminar
                break

            # Passo 5: determinar o novo conjunto de trabalho
            searchDir = saida_svm - self.Y
            set1 = np.logical_and(np.logical_or(self.SV, self.class0), np.logical_or(np.logical_not(self.SVBound), self.class1))
            set2 = np.logical_and(np.logical_or(self.SV, self.class1), np.logical_or(np.logical_not(self.SVBound), self.class0))

            if self.randomWS:
                searchDir = np.random.rand(self.N, 1)
                set1 = self.class1
                set2 = self.class0
                self.randomWS = False


            # Passo 6: Seleciona o working set
            #          (QPsize/2 exemplos de set1, QPsize/2 de set2
            worksetOld = np.copy(workset)
            workset = np.full((self.N, 1), False)

            if np.flatnonzero(np.logical_or(set1, set2)).size <= self.qpsize:
                workset[np.logical_or(set1, set2)] = True
            elif np.flatnonzero(set1).size <= math.floor(self.qpsize/2):
                workset[set1] = True
                set2 = np.flatnonzero(np.logical_and(set2, np.logical_not(workset)))
                ind = searchDir[set2].argsort(0)
                from2 = min(set2.size, self.qpsize - np.flatnonzero(workset).size)
                workset[set2[ind[:from2]]] = True
            elif np.flatnonzero(set2).size <= math.floor(self.qpsize/2):
                workset[set2] = True
                set1 = np.flatnonzero(np.logical_and(set1, np.logical_not(workset)))
                ind = -searchDir[set1].argsort(0)
                from1 = min(set1.size, self.qpsize - np.flatnonzero(workset).size)
                workset[set1[ind[:from1]]] = True
            else:
                set1 = np.flatnonzero(set1)
                ind = -searchDir[set1].argsort(0)
                from1 = min(set1.size, math.floor(self.qpsize /2))
                workset[set1[ind[:from1]]] = True

                set2 = np.flatnonzero(np.logical_and(set2, np.logical_not(workset)))
                ind = searchDir[set2].argsort(0)
                from2 = min(set2.size, self.qpsize - np.flatnonzero(workset).size)
                workset[set2[ind[:from2]]] = True

            worksetind = np.flatnonzero(workset)

            if np.all(workset==worksetOld):
                sameWS +=1
                if sameWS == 3:
                    break

            worksize = worksetind.size
            nonworkset = np.logical_not(workset)

            # Passo 7: Determine a parte da programação linear
            nonworkSV = np.flatnonzero(np.logical_and(nonworkset, self.SV))
            qBN = 0
            if nonworkSV.size > 0:
                chunks = math.ceil(nonworkSV.size/self.chunksize)
                for ch in range(chunks):
                    ind_ini = self.chunksize*ch
                    ind_fim = min(nonworkSV.size, self.chunksize*(ch+1))
                    Ki = self.calc_rbf(X[worksetind, :], X[nonworkSV[ind_ini:ind_fim], :])
                    qBN += Ki.dot(self.alpha[nonworkSV[ind_ini:ind_fim]] * Y[nonworkSV[ind_ini:ind_fim]])
                qBN = qBN * Y[workset].reshape(-1,1)

            f = qBN - np.ones((worksize,1))

            # Passo 8: Soluciona a programação quadrática
            H = self.calc_rbf(self.X[worksetind, :], self.X[worksetind, :])

            H += np.diag(np.ones((worksize, 1))*np.spacing(1)**(2/3))

            H = H * (self.Y[workset].dot(Y[workset].T))
            A = Y[workset].reshape(1,-1).astype('float')

            if nonworkSV.size > 0:
                eqconstr = -self.alpha[nonworkSV].T.dot(Y[nonworkSV])
            else:
                eqconstr = np.zeros(1)

            VLB = np.zeros((1, worksize))
            VUB = self.C[workset].astype('float')


            start_val = self.alpha[workset]
            start_val = start_val.reshape(worksize, 1)

            # cvxopt = quadprog(C1, C2, C3, C4, C5, C6      , C7 , C8 ,
            # matlab = quadprog(H , f , [], [], A , eqconstr, VLB, VUB, startVal,

            # _H = matrix(H) #P #todo: ver direito isso ae
            _f = matrix(f) #q

            tmp1 = np.diag(np.ones(worksize) * -1)
            tmp2 = np.identity(worksize)
            G = matrix(np.vstack((tmp1, tmp2)))

            tmp1 = np.zeros(worksize)
            tmp2 = np.ones(worksize) * 10.
            h = matrix(np.hstack((tmp1, tmp2)))

            _c3 = matrix(np.vstack((np.eye(worksize)*-1,np.eye(worksize)))) #G
            _c4 = matrix(np.hstack((np.zeros(worksize), np.ones(worksize) * 10))) #h
            _A = matrix(A) #A
            _eqconstr = matrix(eqconstr) #b
            _VLB = matrix(VLB)
            _VUB = matrix(VUB)
            _start_val = matrix(start_val)

            #todo: melhorar isso
            # ref: https://python.plainenglish.io/introducing-python-package-cvxopt-implementing-svm-from-scratch-dc40dda1da1f
            teste = Y[workset]
            teste = teste.reshape(-1, 1)
            z = teste * X[worksetind, :]
            oh = matrix(np.dot(z, z.T).astype('float'))

            solvers.options['maxiters'] = 1000
            solvers.options['show_progress'] = self.debug
            solvers.options['abstol'] = 1e-8
            solvers.options['reltol'] = 1e-8
            solvers.options['feastol'] = 1e-8
            solvers.options['refinement'] = 1
            sol = solvers.qp(oh, _f, G, h, A=_A, b=_eqconstr, initvals=_start_val)
            workAlpha = np.array(sol['x'])

            alphaOld = np.copy(self.alpha)
            self.alpha[workset] = workAlpha.squeeze()
            iteracao += 1

        self.svcoeff = self.alpha[self.svind] * Y[self.svind]
        self.SV = X[self.svind, :]

    def findSV(self):
        maxalpha = self.alpha.max()
        if maxalpha > self.alphatol:
            self.SVThresh = self.alphatol
        else:
            self.SVThresh = math.exp((math.log(max(np.spacing(1), maxalpha))+ math.log(np.spacing(1)))/2)

        self.SV = self.alpha>=self.SVThresh

        self.SVBound = self.alpha>=(self.C-self.alphatol)

        self.SVnonBound = np.logical_and(self.SV, np.logical_not(self.SVBound))
        self.svind = np.flatnonzero(self.SV)

    def calc_rbf(self, X1, X2):
        N1, d = X1.shape
        N2, d = X2.shape

        dist2 = np.tile(np.sum((X1 ** 2).T, 0), (N2, 1)).T
        dist2 += np.tile(np.sum((X2 ** 2).T, 0), (N1, 1))
        dist2 -= 2 * X1.dot(X2.T)
        return np.exp(-dist2/2)

    def return_instance_for_predict(self):
        # limpando dados para diminuir o tamanho
        self.X = None
        self.Y = None
        self.SV = self.SV.astype(np.float16)
        return self

    def calc_saida(self, X):
        N, d = X.shape
        nbSV = self.SV.shape[0]
        chsize = self.chunksize
        Y1 = np.zeros((N, 1))
        chunks1 = math.ceil(N / chsize)
        chunks2 = math.ceil(nbSV / chsize)

        for ch1 in range(chunks1):
            ini_ind1 = (ch1) * self.chunksize
            fim_ind1 = min(self.N, (ch1 + 1) * self.chunksize)
            for ch2 in range(chunks2):
                ini_ind2 = (ch2) * self.chunksize
                fim_ind2 = min(nbSV, (ch2 + 1) * self.chunksize)
                K12 = self.calc_rbf(X[ini_ind1:fim_ind1, :], self.SV[ini_ind2:fim_ind2, :])
                Y1[ini_ind1:fim_ind1] += K12.dot(self.svcoeff[ini_ind2:fim_ind2])

        Y1 += self.bias
        Y = np.sign(Y1)
        Y[Y==0] = 1
        return Y, Y1


def split_data(dataset, id, train_percent, validation_percent=0.):
  dataset = dataset[np.where(dataset[:,0]==id)]
  train_registers = int(dataset.shape[0] * train_percent)
  test_registers = dataset.shape[0] - train_registers
  if test_registers == 0 or train_registers == 0:
    print("Id {} com registros insuficientes.".format(id))
    x_train = dataset[0:0, 1:]
    y_train = dataset[0:0, [0]]
    x_test = dataset[0:0, 1:]
    y_test = dataset[0:0, [0]]
  else:
    x_train = dataset[0:train_registers, 1:]
    y_train = dataset[0:train_registers, [0]]
    x_test = dataset[train_registers:, 1:]
    y_test = dataset[train_registers:, [0]]

  return x_train, y_train, x_test, y_test

def split_data_ovo(dataset, id, train_percent, validation_percent=0.):
  dataset = dataset[np.where(dataset[:,0]==id)]
  train_registers = int(dataset.shape[0] * train_percent)
  test_registers = dataset.shape[0] - train_registers
  if test_registers == 0 or train_registers == 0:
    print("Id {} com registros insuficientes.".format(id))
    x_train = dataset[0:0, 1:]
    x_test = dataset[0:0, 1:]
    x_val = dataset[0:0, 1:]
  else:
    x_train = dataset[0:train_registers, 1:]
    x_test = dataset[train_registers:, 1:]
    x_val = dataset[0:0, 1:]
  return x_train, x_val, x_test


def generate_OvO_pairs(list_of_classes):
    # deixa distinct e ordenado
    list_of_classes = list(set(list_of_classes))
    list_ovo_pairs = []
    count_classes = len(list_of_classes)
    for i in range(count_classes - 1):
        ini_rest = i + 1
        one_item = list_of_classes[i]
        rest = list_of_classes[ini_rest:]
        for rest_item in rest:
            list_ovo_pairs.append([one_item, rest_item])

    return list_ovo_pairs

def generate_dictionary_dataset(path_dataset_hdf5, percent_train=0.6, percent_validation=0.2, total_classes=-1):
    start_time = datetime.datetime.now()
    dict_train = {}
    dict_validation = {}
    dict_test = {}

    f = h5py.File(path_dataset_hdf5, 'r')
    hog_array = cp.array(f['descriptor'])  # cupy array

    classes = np.array(cp.unique(hog_array[:, 0]).get())
    random.seed(2)
    random.shuffle(classes)

    if total_classes > -1:
        classes= classes[:total_classes]


    for classe in classes:
        # concatena os conjuntos
        x_tr, x_val, x_te = split_data_ovo(hog_array, classe, percent_train, percent_validation)
        dict_train[classe] = np.array(x_tr.get())
        dict_validation[classe] = np.array(x_val.get())
        dict_test[classe] = np.array(x_te.get())
    end_time = datetime.datetime.now()
    print("Tempo para criar os dicionarios: {}".format(end_time-start_time))
    return dict_train, dict_validation, dict_test


def train_hog(path_hog):
    time_start = datetime.datetime.now()
    f = h5py.File(path_hog, 'r')

    hog_array = np.array(f['descriptor'])  # numpy array

    qtd_classes = 2
    percent_train = 0.6

    random.seed(2)
    classes = np.unique(hog_array[:, 0])
    # random.shuffle(classes)
    classes = classes[0: qtd_classes]
    classes = np.array(classes)
    # inicializa com o id 0
    x_train_hog, y_train_hog, x_test_hog, y_test_hog = split_data(hog_array, 0, percent_train)

    for classe in classes:
        # concatena os conjuntos
        x_tr, y_tr, x_te, y_te = split_data(hog_array, classe, percent_train)
        x_train_hog = np.concatenate([x_train_hog, x_tr])
        y_train_hog = np.concatenate([y_train_hog, y_tr])
        x_test_hog = np.concatenate([x_test_hog, x_te])
        y_test_hog = np.concatenate([y_test_hog, y_te])

    train_x, train_y, test_x, test_y = x_train_hog, y_train_hog, x_test_hog, y_test_hog


    train_y[train_y == 1] = 1
    train_y[train_y == 2] = -1

    test_y[test_y == 1] = 1
    test_y[test_y == 2] = -1

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    svm = SVM(train_x, train_y, 10 ** -2)
    svm.prep()
    svm.trainSVM()

    Ysvm, Y1svm = svm.calc_saida(train_x)

    pos = np.flatnonzero(train_y == 1)
    neg = np.flatnonzero(train_y == -1)

    TP = np.sum(Ysvm[pos] == 1)
    FN = np.sum(Ysvm[pos] == -1)
    TN = np.sum(Ysvm[neg] == -1)
    FP = np.sum(Ysvm[neg] == 1)

    precisao = TP / (TP + FP)
    recall = TN / (TN + FN)
    acuracia = (TP + TN) / (TP + TN + FP + FN)

    print("Treino - Precisao: {} Recall:{} Acuracia:{}".format(precisao, recall, acuracia))

    # Teste

    Ysvm, Y1svm = svm.calc_saida(test_x)

    pos = np.flatnonzero(test_y == 1)
    neg = np.flatnonzero(test_y == -1)

    TP = np.sum(Ysvm[pos] == 1)
    FN = np.sum(Ysvm[pos] == -1)
    TN = np.sum(Ysvm[neg] == -1)
    FP = np.sum(Ysvm[neg] == 1)

    precisao = TP / (TP + FP)
    recall = TN / (TN + FN)
    acuracia = (TP + TN) / (TP + TN + FP + FN)



    time_end = datetime.datetime.now()
    print("Tempo total: {}".format(time_end-time_start))
    print("Teste - Precisao: {} Recall:{} Acuracia:{}".format(precisao, recall, acuracia))


def train_lbp(path_lbp):

    f = h5py.File(path_lbp, 'r')

    hog_array = np.array(f['descriptor'])  # numpy array

    qtd_classes = 2
    percent_train = 0.6

    random.seed(2)
    classes = np.unique(hog_array[:, 0])
    random.shuffle(classes)
    classes = classes[0: qtd_classes]
    classes = np.array(classes)
    # inicializa com o id 0
    x_train_hog, y_train_hog, x_test_hog, y_test_hog = split_data(hog_array, 0, percent_train)

    for classe in classes:
        # concatena os conjuntos
        x_tr, y_tr, x_te, y_te = split_data(hog_array, classe, percent_train)
        x_train_hog = np.concatenate([x_train_hog, x_tr])
        y_train_hog = np.concatenate([y_train_hog, y_tr])
        x_test_hog = np.concatenate([x_test_hog, x_te])
        y_test_hog = np.concatenate([y_test_hog, y_te])

    train_x, train_y, test_x, test_y = x_train_hog, y_train_hog, x_test_hog, y_test_hog

    # normalizando para valores
    train_x = train_x / 255.
    test_x = test_x / 255.

    train_y[train_y == 1974] = 1
    train_y[train_y == 1937] = -1

    test_y[test_y == 1974] = 1
    test_y[test_y == 1937] = -1

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    time_start = datetime.datetime.now()

    svm = SVM(train_x, train_y, 10 ** -2)
    svm.debug = False
    svm.prep()
    svm.trainSVM()

    time_end = datetime.datetime.now()

    Ysvm, Y1svm = svm.calc_saida(train_x)

    pos = np.flatnonzero(train_y == 1)
    neg = np.flatnonzero(train_y == -1)

    TP = np.sum(Ysvm[pos] == 1)
    FN = np.sum(Ysvm[pos] == -1)
    TN = np.sum(Ysvm[neg] == -1)
    FP = np.sum(Ysvm[neg] == 1)

    precisao = TP / (TP + FP)
    recall = TN / (TN + FN)
    acuracia = (TP + TN) / (TP + TN + FP + FN)

    print("Treino Precisao: {} Recall:{} Acuracia:{}".format(precisao, recall, acuracia))

    # Teste

    Ysvm, Y1svm = svm.calc_saida(test_x)

    pos = np.flatnonzero(test_y == 1)
    neg = np.flatnonzero(test_y == -1)

    TP = np.sum(Ysvm[pos] == 1)
    FN = np.sum(Ysvm[pos] == -1)
    TN = np.sum(Ysvm[neg] == -1)
    FP = np.sum(Ysvm[neg] == 1)

    precisao = TP / (TP + FP)
    recall = TN / (TN + FN)
    acuracia = (TP + TN) / (TP + TN + FP + FN)

    print("Teste - Precisao: {} Recall:{} Acuracia:{}".format(precisao, recall, acuracia))


    print("Tempo total: {}".format(time_end - time_start))

# train_hog("./data/hog_11_15_20_56")
# train_lbp("./data/lbp_grid_total")
#
# unique_ids = [x for x in range(2000)]
# result = generate_OvO_pairs(unique_ids)

def train_ovo_lbp(total_classes=4):
    time_start = datetime.datetime.now()
    dict_ovo_weights = {}
    dict_train, dict_validation, dict_test = generate_dictionary_dataset("./data/lbp_grid_total", total_classes=total_classes)

    lista_classes = np.array(list(dict_train.keys()))
    list_ovo = generate_OvO_pairs(lista_classes)
    logs = []
    logs.append("=============================================================")
    logs.append("Execução em: {}".format(datetime.datetime.now()))
    logs.append("Quantidade de SVMs One-vs-One: {}".format(len(list_ovo)) )
    logs.append("Quantidade de Artistas: {}".format(len(lista_classes)))
    logs.append("=============================================================")

    for ovo in list_ovo:
        x_train = np.concatenate([dict_train[ovo[0]],dict_train[ovo[1]]])
        y_train = np.concatenate([np.full(dict_train[ovo[0]].shape[0], -1),np.full(dict_train[ovo[1]].shape[0], 1)]).reshape(-1,1)
        x_train = x_train / 255
        svm = SVM(x_train, y_train)

        logs.append("Classes OvO: {}".format(ovo))
        logs.append("Shape do array de treino: {}".format( x_train.shape))

        svm.debug = False
        svm.prep()
        svm.trainSVM()


        dict_ovo_weights[tuple(ovo)] = svm.return_instance_for_predict()

    time_end = datetime.datetime.now()
    print("Tempo train_ovo_lbp: {}".format(time_end - time_start))
    logs.append("Tempo train_ovo_lbp: {}".format(time_end - time_start))
    pickle.dump(dict_ovo_weights, open("5016_svm_lbp_{}_.dat".format(total_classes), "wb"))
    return dict_test, dict_ovo_weights, logs

def train_ovo_hog(total_classes=4):
    logs = []
    time_start = datetime.datetime.now()
    dict_ovo_weights = {}
    dict_train, dict_validation, dict_test = generate_dictionary_dataset("./data/hog_11_15_20_56", total_classes=total_classes)
    list_ovo = generate_OvO_pairs(dict_train.keys())

    logs.append("=============================================================")
    logs.append("Execução em: {}".format(datetime.datetime.now()))
    logs.append("Quantidade de SVMs One-vs-One: {}".format(len(list_ovo)))
    logs.append("Quantidade de Artistas: {}".format(len(dict_train.keys())))
    logs.append("=============================================================")
    loop_start = datetime.datetime.now()
    iter = 0
    for ovo in list_ovo:
        x_train = np.concatenate([dict_train[ovo[0]],dict_train[ovo[1]]])
        y_train = np.concatenate([np.full(dict_train[ovo[0]].shape[0], -1),np.full(dict_train[ovo[1]].shape[0], 1)]).reshape(-1,1)
        svm = SVM(x_train, y_train)

        logs.append("Classes OvO: {}".format(ovo))
        logs.append("Shape do array de treino: {}".format(x_train.shape))

        svm.debug = False
        svm.prep()
        svm.trainSVM()

        dict_ovo_weights[tuple(ovo)] = svm.return_instance_for_predict()

        if iter % 1000 == 0:
            loop_end = datetime.datetime.now()
            logs.append("Tempo para treinar 1000 SVMs: {}".format(loop_end-loop_start))
            print("Tempo para treinar 1000 SVMs: {}".format(loop_end-loop_start))
            loop_start = datetime.datetime.now()

        iter += 1

    time_end = datetime.datetime.now()
    print("Tempo train_ovo_hog: {}".format(time_end - time_start))
    logs.append("Tempo train_ovo_hog: {}".format(time_end - time_start))
    pickle.dump(dict_ovo_weights, open("5016_svm_hog_{}_.dat".format(total_classes), "wb"))
    return dict_test, dict_ovo_weights, logs

def evaluate_ovo_weights_hog(dict_test, dict_ovo_weights, logs):
    time_start = datetime.datetime.now()
    tests = list(dict_test.keys())
    dataset_x_test = dict_test[tests[0]]
    dataset_y_test = np.full(dict_test[tests[0]].shape[0], tests[0]).reshape(-1,1)

    for i in range(1,len(tests)):
        dataset_x_test = np.concatenate([dataset_x_test, dict_test[tests[i]]])
        dataset_y_test = np.concatenate([dataset_y_test, np.full(dict_test[tests[i]].shape[0], tests[i]).reshape(-1,1)])

    ovos = dict_ovo_weights.keys()
    votos = [[x, []] for x in range(dataset_y_test.shape[0])]

    for ovo in ovos:
        Ysvm, Y1svm = dict_ovo_weights[ovo].calc_saida(dataset_x_test)
        id = 0
        for y in Ysvm:
            if y >= 0:
                classe = ovo[1]
            else:
                classe = ovo[0]
            votos[id][1].append(classe)
            id += 1

    y_alcancado = []
    for voto in votos:
        pos, count = np.unique(voto[1], return_counts=True)
        y_alcancado.append([pos[np.argmax(count)]])

    y_alcancado = np.array(y_alcancado)
    acuracia = (y_alcancado == dataset_y_test).astype('int').mean()

    print("Acuracia hog: {}".format(acuracia))
    logs.append("Acuracia hog: {}".format(acuracia))
    time_end = datetime.datetime.now()
    logs.append("Tempo evaluate_ovo_weights_hog: {}".format(time_end - time_start))
    print("Tempo evaluate_ovo_weights_hog: {}".format(time_end - time_start))
    pickle.dump(logs, open("5016_svm_hog_logs.pkl", "wb"))

def evaluate_ovo_weights_lbp(dict_test, dict_ovo_weights, logs):
    time_start = datetime.datetime.now()
    tests = list(dict_test.keys())
    dataset_x_test = dict_test[tests[0]]
    dataset_y_test = np.full(dict_test[tests[0]].shape[0], tests[0]).reshape(-1, 1)

    for i in range(1, len(tests)):
        dataset_x_test = np.concatenate([dataset_x_test, dict_test[tests[i]]])
        dataset_y_test = np.concatenate(
            [dataset_y_test, np.full(dict_test[tests[i]].shape[0], tests[i]).reshape(-1, 1)])

    ovos = dict_ovo_weights.keys()
    votos = [[x, []] for x in range(dataset_y_test.shape[0])]
    dataset_x_test = dataset_x_test / 255
    for ovo in ovos:
        Ysvm, Y1svm = dict_ovo_weights[ovo].calc_saida(dataset_x_test)
        id = 0
        for y in Ysvm:
            if y >= 0:
                classe = ovo[1]
            else:
                classe = ovo[0]
            votos[id][1].append(classe)
            id += 1

    y_alcancado = []
    for voto in votos:
        pos, count = np.unique(voto[1], return_counts=True)
        y_alcancado.append([pos[np.argmax(count)]])

    y_alcancado = np.array(y_alcancado)
    acuracia = (y_alcancado == dataset_y_test).astype('int').mean()

    print("Acuracia lbp: {}".format(acuracia))
    logs.append("Acuracia lbp: {}".format(acuracia))
    time_end = datetime.datetime.now()
    print("Tempo evaluate_ovo_weights_lbp: {}".format(time_end - time_start))
    logs.append("Tempo evaluate_ovo_weights_lbp: {}".format(time_end - time_start))
    pickle.dump(logs, open("5016_svm_lbp_logs.pkl", "wb"))

#
# dict_test, dict_ovo_weights, logs = train_ovo_hog(100)
# evaluate_ovo_weights_hog(dict_test, dict_ovo_weights, logs)
#
# dict_test, dict_ovo_weights, logs = train_ovo_lbp(100)
# evaluate_ovo_weights_lbp(dict_test, dict_ovo_weights, logs)

def train_hog_full(total_classes=4):
    logs = []
    time_start = datetime.datetime.now()
    dict_ovo_weights = {}
    dict_train, dict_validation, dict_test = generate_dictionary_dataset("./data/hog_11_15_20_56", total_classes=total_classes)
    for x in dict_train.keys():
        print(x)

    for ovo in list_ovo:
        x_train = np.concatenate([dict_train[ovo[0]],dict_train[ovo[1]]])
        y_train = np.concatenate([np.full(dict_train[ovo[0]].shape[0], -1),np.full(dict_train[ovo[1]].shape[0], 1)]).reshape(-1,1)
        svm = SVM(x_train, y_train)

        logs.append("Classes OvO: {}".format(ovo))
        logs.append("Shape do array de treino: {}".format(x_train.shape))

        svm.debug = False
        svm.prep()
        svm.trainSVM()


# train_hog_full(10)
dict_moc = {}
def processa_moc(total_classes=5, ecoc=1, path_hdf="", lbp=False):
    # quebra o dataset em treino ,teste e validacao
    dict_train, dict_validation, dict_test = generate_dictionary_dataset(path_hdf,
                                                                         total_classes=total_classes)
    # lista de classes unicas
    unique_classes = list(dict_train.keys())

    # gera uma matriz de binários para atribuir às classes
    moc_qty = math.ceil(math.log2(len(unique_classes)) * ecoc)

    binary_format = "{0:0" + str(moc_qty) + "b}"
    matriz_binarios = []
    # lista_randomica
    max_random_val = 2 ** moc_qty

    if ecoc > 1:
        randomlist = []
        for i in range(len(unique_classes)):
            n = random.randint(0, max_random_val)
            randomlist.append(n)
    else:
        randomlist = [x for x in range(max_random_val)]



    for i in randomlist:
        bin_rep = binary_format.format(i)
        matriz_binarios.append(bin_rep)

    # embaralha a matriz, para evitar desbalanceamento em caso de
    # conjuntos com (2^x)+ 1
    if ecoc==1:
        random.shuffle(matriz_binarios)

    dict_moc = {}
    for i in range(len(unique_classes)):
        dict_moc[unique_classes[i]] = matriz_binarios[i]

    svm_datasets = []
    for i in range(moc_qty):
        classes_dataset_0 = []
        classes_dataset_1 = []
        for classe in dict_moc.keys():
            if dict_moc[classe][i] == "1":
                classes_dataset_1.append(classe)
            else:
                classes_dataset_0.append(classe)
        svm_datasets.append((classes_dataset_0, classes_dataset_1))

    print("fim")
    return dict_moc, svm_datasets

def get_x_y(dict_data, lista_classe_0, lista_classe_1, lbp=False):
    x_train_0 = dict_data[lista_classe_0[0]]
    y_classes_reais_0 = np.full(x_train_0.shape[0], lista_classe_0[0])
    for i in range(1, len(lista_classe_0)):
        x_train_0 = np.concatenate([x_train_0, dict_data[lista_classe_0[i]]])
        y_classes_reais_0 = np.concatenate([y_classes_reais_0, np.full(dict_data[lista_classe_0[i]].shape[0], lista_classe_0[0])])

    x_train_1  = dict_data[lista_classe_1[0]]
    y_classes_reais_1 = np.full(x_train_1.shape[0], lista_classe_1[0])
    for i in range(1, len(lista_classe_1)):
        x_train_1 = np.concatenate([x_train_1, dict_data[lista_classe_1[i]]])
        y_classes_reais_1 = np.concatenate([y_classes_reais_1, np.full(dict_data[lista_classe_1[i]].shape[0], lista_classe_1[0])])

    x_train = np.concatenate([x_train_0, x_train_1])
    y_train_0 = np.full(x_train_0.shape[0], -1)
    y_train_1 = np.full(x_train_1.shape[0], 1)
    y_train = np.concatenate([y_train_0, y_train_1])
    y_classes_reais = np.concatenate([y_classes_reais_0, y_classes_reais_1])
    y_train =  y_train.reshape(-1,1)
    y_classes_reais = y_classes_reais.reshape(-1,1)
    return x_train, y_train, y_classes_reais

def hamming_distance(bit_1, bit_2):
    hamming = 0
    size = len(bit_1)
    for x in range(size):
        hamming += abs(int(bit_2[x]) - int(bit_1[x]))
    return hamming

def min_hamming_distance(val, list_vals):
    set_hammings = set()
    for j in list_vals:
        set_hammings.add((hamming_distance(val, j), j))
    return set_hammings.pop()[1]



def train_svm_ocs(dict_moc, svm_datasets, total_classes, path_hdf, lbp=False):
    dict_train, dict_validation, dict_test = generate_dictionary_dataset(path_hdf,
                                                                         total_classes=total_classes)

    logs = []
    moc_svms = []
    for svm_dataset in svm_datasets:
        ini_treino = datetime.datetime.now()
        x_train, y_train, y_classes_reais = get_x_y(dict_train, svm_dataset[0], svm_dataset[1])
        if lbp:
            x_train = x_train / 255
        svm_ = SVM(x_train, y_train)


        # teste = SVC()
        # teste.fit(x_train, y_train)
        # teste_y = teste.predict(x_train)
        # acc =accuracy_score(y_train, teste_y)

        logs.append("Shape do array de treino: {}".format(x_train.shape))

        svm_.debug = False
        svm_.prep()
        svm_.trainSVM()

        Ysvm, Y1svm = svm_.calc_saida(x_train)

        pos = np.flatnonzero(y_train == 1)
        neg = np.flatnonzero(y_train == -1)

        TP = np.sum(Ysvm[pos] == 1)
        FN = np.sum(Ysvm[pos] == -1)
        TN = np.sum(Ysvm[neg] == -1)
        FP = np.sum(Ysvm[neg] == 1)

        # precisao = TP / (TP + FP)
        # recall = TN / (TN + FN)
        acuracia = (TP + TN) / (TP + TN + FP + FN)
        fim_treino = datetime.datetime.now()
        print("Treino Acuracia:{} tempo:{}".format(acuracia, fim_treino-ini_treino))
        moc_svms.append(svm_.return_instance_for_predict())

    resultados = []
    for svm_treinado in moc_svms:
        Ysvm, Y1svm = svm_treinado.calc_saida(x_train)
        Ysvm[Ysvm < 0] = '0'
        Ysvm[Ysvm > 0] = '1'
        resultados.append(Ysvm.astype(int).astype(str))


    # cria um dicionário inverso para o
    dict_classes = {}
    for classe in dict_moc.keys():
        dict_classes[dict_moc[classe]] = classe


    res_moc = []
    for res in range(len(resultados[0])):
        lista_res = "".join([res_y[res][0] for res_y in resultados])
        res_moc.append(lista_res)

    res_moc_classes = []

    for x in res_moc:
        if x in dict_classes.keys():
            res_moc_classes.append(dict_classes[x])
        else:
            res_moc_classes.append(dict_classes[min_hamming_distance(x, dict_classes.keys())])

    res_moc_classes = np.array(res_moc_classes).reshape(-1,1)


    corretos = 0
    idx = 0
    for x in range(res_moc_classes.shape[0]):
        if res_moc_classes[x][0] == y_classes_reais[idx][0]:
            corretos += 1
        idx += 1

    print("treino corretos", (res_moc_classes==y_classes_reais).astype(int).sum())
    print("treino acuracia", (res_moc_classes==y_classes_reais).astype(int).mean())

    pickle.dump(moc_svms, open("5016_svm_hog_moc.dat".format(total_classes), "wb"))

    unique_ids = list(dict_test.keys())
    x_test = dict_test[unique_ids[0]]
    y_test = np.full(x_test.shape[0], unique_ids[0])
    for x in range(1, len(unique_ids)):
        x_test = np.concatenate([x_test, dict_test[unique_ids[x]]])
        y_test = np.concatenate([y_test, np.full(dict_test[unique_ids[x]].shape[0], unique_ids[x])])

    resultados = []
    for svm_treinado in moc_svms:
        if lbp:
            x_test = x_test / 255
        Ysvm, Y1svm = svm_treinado.calc_saida(x_test)
        Ysvm[Ysvm < 0] = 0
        Ysvm[Ysvm > 0] = 1
        resultados.append(Ysvm.astype(int).astype(str))

    res_moc = []
    for res in range(len(resultados[0])):
        lista_res = [res_y[res][0] for res_y in resultados]
        lista_res = [str(x) for x in lista_res]
        res_moc.append("".join(lista_res))

    res_moc_classes = []

    for x in res_moc:
        if x in dict_classes.keys():
            res_moc_classes.append(dict_classes[x])
        else:
            res_moc_classes.append(dict_classes[min_hamming_distance(x, dict_classes.keys())])

    res_moc_classes = np.array(res_moc_classes).reshape(-1, 1)

    corretos = 0
    idx = 0
    ## melhorar comparação
    for x in range(res_moc_classes.shape[0]):
        if res_moc_classes[idx][0] == y_test[idx]:
            corretos += 1
        idx += 1

    print("teste corretos", corretos)
    print("teste acuracia", corretos/y_test.size)

    return moc_svms



dict_moc, svm_datasets = processa_moc(64, 1, "./data/hog_11_15_20_56")
train_svm_ocs(dict_moc, svm_datasets, 64, "./data/hog_11_15_20_56")


dict_moc, svm_datasets = processa_moc(64, 1, "./data/lbp_grid_total", lbp=True)
train_svm_ocs(dict_moc, svm_datasets, 64, "./data/lbp_grid_total", lbp=True)

