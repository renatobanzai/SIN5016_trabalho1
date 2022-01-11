import logging
import numpy as np
import cupy as cp
import math
from cvxopt import matrix, solvers

class svm_gpu:
    def __init__(self, X, Y, config=None):
        '''
        inicializador da classe
        :param X: caracteristicas para treinar
        :param Y: resultado desejado
        :param log_file: nome do arquivo de log para guardar
        '''
        logging.info("instanciado um svm")
        self.X = X
        self.Y = Y
        self.config = config
        if config:
            self.kkttol = config['kkttol']
            self.chunksize = config['chunksize']
            self.bias = config['bias']
            self.sv = config['sv']
            self.svcoeff = config['svcoeff']
            self.normalw = config['normalw']
            self.C = config['C']
            self.h = config['h']
            self.debug = config['debug']
            self.alphatol = config['alphatol']
            self.SVThresh = config['SVThresh']
            self.qpsize = config['qpsize']
            self.logs = config['logs']
            self.configs = config['configs']
            self.kernelpar = config['kernelpar']
            self.randomWS = config['randomWS']
        else:
            self.kkttol = 5e-2
            self.chunksize = 4000
            self.bias = []
            self.sv = []
            self.svcoeff = []
            self.normalw = []
            self.C = 2
            self.h = 0.01
            self.debug = True
            self.alphatol = 1e-2
            self.SVThresh = 0.
            self.qpsize = 512
            self.logs = []
            self.configs = {}
            self.kernelpar = 1  # ao aumentar eentre 1 e 2, melhorou a acuracia em testes
            self.randomWS = True

    def prep(self):
        '''
        etapa de validacao dos atributos
        :return:
        '''
        self.N, self.ne = self.X.shape
        self.class0 = self.Y == -1
        self.class1 = self.Y == 1

        self.Y = self.Y.reshape(self.N, 1)
        self.qpsize = min(self.N, self.qpsize)
        self.C = cp.full((self.N, 1), self.C)
        self.alpha = cp.zeros((self.N, 1))

        if self.debug:
            logging.debug("Working set possui {} exemplos de classe positiva e {} exemplos de classe negativa.".format(self.Y[self.class1].shape[0], self.Y[self.class0].shape[0]))





    def fit(self):
        '''
        Treina o SVM
        codigo obtido na aula da disciplina SIN5016 originalmente em matlab
        convertido para python e orientacao a objetos.
        :return:
        '''
        logging.info("svm.fit")

        X = self.X
        Y = self.Y
        logging.info('X.shape %s', str(X.shape))
        logging.info('Y.shape %s', str(Y.shape))
        logging.info('config: {}'.format(self.config))

        saida_svm = cp.zeros((self.N,1))
        alphaOld = cp.copy(self.alpha)
        if self.Y[self.class1].shape[0] == self.N:
            self.bias = 1
            return

        if self.Y[self.class0].shape[0] == self.N:
            self.bias = -1
            return

        iteracao = 0
        workset = cp.full((self.N,1), False)
        sameWS = 0
        self.bias = 0
        while True:
            # logging.info('Iteracao %i', iteracao)

            # Passo 1: determina os vetores de suporte
            self.findSV()

            # Passo 2: Encontra a saída para o SVM
            if iteracao == 0:
                changedSV = self.svind # todo: cp.copy(self.svind)
                changedAlpha = self.alpha[changedSV] # todo: cp.copy(self.alpha[changedSV])
                saida_svm = cp.zeros((self.N, 1))
            else:
                changedSV = cp.flatnonzero(alphaOld != self.alpha)
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
                    saida_svm[ini_ind1:fim_ind1] = saida_svm[ini_ind1:fim_ind1] + cp.dot(K12, coeff)

            # Passo 3: Calcule o bias da função de decisão
            workSV = cp.flatnonzero(cp.logical_and(self.SVnonBound, workset))
            if not workSV.size == 0:
                self.bias = cp.mean(Y[workSV] - saida_svm[workSV])


            # Passo 4: Calcula as condicoes de KKT
            KKT = (saida_svm+self.bias)*self.Y-1
            KKTViolations1 = cp.logical_and(self.SVnonBound, (abs(KKT)>self.kkttol))
            KKTViolations2 = cp.logical_and(self.SVBound, KKT > self.kkttol)
            KKTViolations3 = cp.logical_and(cp.logical_not(self.SV), (KKT < (self.kkttol*-1)))
            KKTViolations = cp.logical_or(KKTViolations1, KKTViolations2)
            KKTViolations = cp.logical_or(KKTViolations, KKTViolations3)

            # (uint8((SVnonbound & (abs(KKT) > svm.kkttol)) | ...
            # (SVbound & (KKT > svm.kkttol)) | ...
            #        (~SV & (KKT < -svm.kkttol))))



            count_kkt = len(cp.flatnonzero(KKTViolations))



            if iteracao % 100 == 0:
                logging.info('iteracao: {} KKT violacoes: {}'.format(iteracao, count_kkt))

            if count_kkt == 0:
                # sem violacoes, terminar
                logging.info('fim do treino por fim das violacoes de KKT, total iteracoes: {}'.format(iteracao))
                break

            # Passo 5: determinar o novo conjunto de trabalho
            searchDir = saida_svm - self.Y
            set1 = cp.logical_and(cp.logical_or(self.SV, self.class0), cp.logical_or(cp.logical_not(self.SVBound), self.class1))
            set2 = cp.logical_and(cp.logical_or(self.SV, self.class1), cp.logical_or(cp.logical_not(self.SVBound), self.class0))

            if self.randomWS:
                cp.random.seed(0)
                searchDir = cp.random.rand(self.N, 1)
                set1 = self.class1
                set2 = self.class0
                self.randomWS = False


            # Passo 6: Seleciona o working set
            #          (QPsize/2 exemplos de set1, QPsize/2 de set2
            worksetOld = cp.copy(workset)
            workset = cp.full((self.N, 1), False)

            if cp.flatnonzero(cp.logical_or(set1, set2)).size <= self.qpsize:
                workset[cp.logical_or(set1, set2)] = True
            elif cp.flatnonzero(set1).size <= math.floor(self.qpsize/2):
                workset[set1] = True
                set2 = cp.flatnonzero(cp.logical_and(set2, cp.logical_not(workset)))
                ind = searchDir[set2].argsort(0)
                from2 = min(set2.size, self.qpsize - cp.flatnonzero(workset).size)
                workset[set2[ind[:from2]]] = True
            elif cp.flatnonzero(set2).size <= math.floor(self.qpsize/2):
                workset[set2] = True
                set1 = cp.flatnonzero(cp.logical_and(set1, cp.logical_not(workset)))
                ind = -searchDir[set1].argsort(0)
                from1 = min(set1.size, self.qpsize - cp.flatnonzero(workset).size)
                workset[set1[ind[:from1]]] = True
            else:
                set1 = cp.flatnonzero(set1)
                ind = (-searchDir[set1]).argsort(0)
                from1 = min(set1.size, math.floor(self.qpsize /2))
                workset[set1[ind[:from1]]] = True

                set2 = cp.flatnonzero(cp.logical_and(set2, cp.logical_not(workset)))
                ind = searchDir[set2].argsort(0)
                from2 = min(set2.size, self.qpsize - cp.flatnonzero(workset).size)
                workset[set2[ind[:from2]]] = True

            worksetind = cp.flatnonzero(workset)



            if cp.all(workset==worksetOld):
                sameWS +=1
                if sameWS == 3:
                    logging.info('fim do treino por por permanecer no mesmo workingset por: %i iteracoes', sameWS)
                    break
            else:
                sameWS = 0

            worksize = worksetind.size
            nonworkset = cp.logical_not(workset)

            # Passo 7: Determine a parte da programação linear
            nonworkSV = cp.flatnonzero(cp.logical_and(nonworkset, self.SV))
            qBN = 0
            if nonworkSV.size > 0:
                chunks = math.ceil(nonworkSV.size/self.chunksize)
                for ch in range(chunks):
                    ind_ini = self.chunksize*ch
                    ind_fim = min(nonworkSV.size, self.chunksize*(ch+1))
                    Ki = self.calc_rbf(X[worksetind, :], X[nonworkSV[ind_ini:ind_fim], :])
                    qBN += Ki.dot(self.alpha[nonworkSV[ind_ini:ind_fim]] * Y[nonworkSV[ind_ini:ind_fim]])
                qBN = qBN * Y[workset].reshape(-1,1)

            f = qBN - cp.ones((worksize,1))

            # Passo 8: Soluciona a programação quadrática
            eps_2_3 = np.spacing(1)**(2/3)
            H = self.calc_rbf(self.X[worksetind, :], self.X[worksetind, :])
            H += cp.diag(cp.ones((worksize, 1))*eps_2_3)
            H = H * (self.Y[workset].dot(Y[workset].T))

            A = Y[workset].T.astype('float').reshape(1,-1)

            if nonworkSV.size > 0:
                eqconstr = -self.alpha[nonworkSV].T.dot(Y[nonworkSV])
            else:
                eqconstr = cp.zeros(1)

            VLB = cp.zeros((1, worksize))
            VUB = self.C[workset].astype('float')


            start_val = self.alpha[workset]
            start_val = start_val.reshape(worksize, 1)

            # cvxopt = quadprog(C1, C2, C3, C4, C5, C6      , C7 , C8 ,
            # matlab = quadprog(H , f , [], [], A , eqconstr, VLB, VUB, startVal,

            _f = matrix(np.array(f.get())) #q

            tmp1 = cp.diag(cp.ones(worksize) * -1)
            tmp2 = cp.identity(worksize)
            G = matrix(np.array(cp.vstack((tmp1, tmp2)).get()))

            tmp1 = cp.zeros(worksize)
            tmp2 = cp.ones(worksize) * 10.
            h = matrix(np.array(cp.hstack((tmp1, tmp2)).get()))

            _c3 = matrix(np.array(cp.vstack((cp.eye(worksize)*-1,cp.eye(worksize))).get())) #G
            _c4 = matrix(np.array(cp.hstack((cp.zeros(worksize), cp.ones(worksize) * 10)).get())) #h
            _A = matrix(np.array(A.get())) #A
            _eqconstr = matrix(np.array(eqconstr.get())) #b
            # _VLB = matrix(VLB)
            # _VUB = matrix(VUB)
            _start_val = matrix(np.array(start_val.get()))

            H = self.calc_rbf(X[worksetind], X[worksetind])
            eps_2_3 = np.spacing(1) ** (2 / 3)
            H = H + cp.diag(cp.full((worksize, worksize), eps_2_3).diagonal())
            H = H * cp.dot(Y[workset].reshape(-1,1), (Y[workset].T.reshape(1,-1)))
            _H = matrix(np.array(H.get()).astype(float))
            solvers.options['maxiters'] = 1000
            solvers.options['show_progress'] = self.debug
            # solvers.options['abstol'] = 1e-8
            # solvers.options['reltol'] = 1e-8
            # solvers.options['feastol'] = 1e-8
            # solvers.options['refinement'] = 1
            sol = solvers.qp(_H, _f, G, h, A=_A, b=_eqconstr, initvals=_start_val)
            workAlpha = cp.array(sol['x'])

            # logging.info('work alpha: %s', str(workAlpha))

            alphaOld = cp.copy(self.alpha)
            self.alpha[workset] = workAlpha.squeeze()
            iteracao += 1

        self.svcoeff = self.alpha[self.svind] * Y[self.svind]
        self.SV = X[self.svind, :]

    def findSV(self):
        '''
        busca os vetores de suporte baseado nos criterios de alphatol / threshold
        :return:
        '''
        maxalpha = self.alpha.max()
        if maxalpha > self.alphatol:
            self.SVThresh = self.alphatol
        else:
            eps_1 = np.spacing(1)
            self.SVThresh = math.exp((math.log(max(eps_1, maxalpha))+ math.log(eps_1))/2)

        self.SV = self.alpha>=self.SVThresh

        self.SVBound = self.alpha>=(self.C-self.alphatol)

        self.SVnonBound = cp.logical_and(self.SV, cp.logical_not(self.SVBound))
        self.svind = cp.flatnonzero(self.SV)

    def calc_rbf(self, X1, X2):
        '''
        funcao kernel rbf
        :param X1:
        :param X2:
        :return:
        '''
        N1, d = X1.shape
        N2, d = X2.shape

        dist2 = cp.tile(cp.sum((X1 ** 2).T, 0), (N2, 1)).T
        dist2 += cp.tile(cp.sum((X2 ** 2).T, 0), (N1, 1))
        dist2 -= 2 * X1.dot(X2.T)
        return cp.exp(-dist2/(2*self.kernelpar**2))

    def return_instance_for_predict(self):
        '''
        Retorna uma instancia do modelo treinado com apenas o necessario
        :return:
        '''
        #
        self.X = None
        self.Y = None
        self.alphatol = None
        self.kkttol = None
        self.alpha = None
        self.svind = None
        self.class1 = None
        self.class0 = None
        self.C = None
        self.normalw = None
        self.ne = None
        self.SVBound = None
        self.SVnonBound = None
        self.SV = self.SV.astype(cp.float16)
        self.svcoeff = self.svcoeff.astype(cp.float16)
        return self

    def predict(self, X):
        N, d = X.shape
        nbSV = self.SV.shape[0]
        chsize = self.chunksize
        Y1 = cp.zeros((N, 1))
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
        Y = cp.sign(Y1)
        Y[Y==0] = 1
        return Y, Y1


def acuracia(y, y_predito):
    '''
    calcula a acuracia
    :param y: y desejado
    :param y_predito: y predito
    :return: acuraria de 0 a 1
    '''
    y.reshape(y_predito.shape)
    acuracia = (y==y_predito).astype(float).mean()
    return acuracia