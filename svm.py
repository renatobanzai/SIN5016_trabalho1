import numpy as np
import math
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, X, Y, alpha0):
        self.X = X
        self.Y = Y
        self.alpha0 = alpha0
        self.kkttol = 5e-2
        self.chunksize = 50
        self.bias = []
        self.sv = []
        self.svcoeff = []
        self.normalw = []
        self.C = 10
        self.h = 0.01
        self.debug = True
        self.alphatol = 1e-2
        self.SVThresh = 0.
        self.qpsize = 6

    def prep(self):
        self.N, self.ne = self.X.shape
        self.class1 = self.Y >= 0
        self.class0 = self.Y < 0

        self.Y[self.class1] = 1;
        self.Y[self.class0] = -1;
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
        alphaOld = self.alpha
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
                svm.bias = np.mean(Y[workSV] - saida_svm[workSV])


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
                np.random.rand(self.N, 1)
                set1 = self.class1
                set2 = self.class0
                self.randomWS = False


            # Passo 6: Seleciona o working set
            #          (QPsize/2 exemplos de set1, QPsize/2 de set2
            worksetOld = workset
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

            _H = matrix(H) #P
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
            solvers.options['show_progress'] = True
            # solvers.options['abstol'] = 1e-8
            # solvers.options['reltol'] = 1e-8
            # solvers.options['feastol'] = 1e-8
            # solvers.options['refinement'] = 1
            sol = solvers.qp(oh, _f, G, h, A=_A, b=_eqconstr, initvals=_start_val)
            workAlpha = np.array(sol['x'])

            alphaOld = self.alpha*1.
            self.alpha[workset] = workAlpha.squeeze()
            iteracao += 1

        self.svcoeff = self.alpha[self.svind] * Y[self.svind]
        self.SV = X[self.svind, :]
        print()

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
        print()

    def calc_rbf(self, X1, X2):
        N1, d = X1.shape
        N2, d = X2.shape

        dist2 = np.tile(np.sum((X1 ** 2).T, 0), (N2, 1)).T
        dist2 += np.tile(np.sum((X2 ** 2).T, 0), (N1, 1))
        dist2 -= 2 * X1.dot(X2.T)
        return np.exp(-dist2/2)

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



X = np.array([[2,7],[3 ,6],[2,2],[8,1],[6,4],[4,8],[9,5],[9,9],[9,4],[6,9],[7,4],[4,4]])
Y = np.array([[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]])

# X = np.array([[2,1],[3,1],[2, 1],[8, 1],[6, 4],[4, 8],[9,5],[9, 9],[9, 4],[6, 9],[7, 4],[4, 4]])
# Y = np.array([[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]])


# X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# Y = np.array([[-1],[1],[1],[-1]])

svm = SVM(X, Y, 10**-2)
svm.prep()
svm.trainSVM()

Ysvm, Y1svm = svm.calc_saida(X)

pos = np.flatnonzero(Y==1)
neg = np.flatnonzero(Y==-1)

TP = np.sum(Ysvm[pos]==1)
FN = np.sum(Ysvm[pos]==-1)
TN = np.sum(Ysvm[neg]==-1)
FP = np.sum(Ysvm[neg]==1)

precisao = TP / (TP+FP)
recall = TN / (TN+FN)
acuracia= (TP+TN)/(TP+TN+FP+FN)

print("Precisao: {} Recall:{} Acuracia:{}".format(precisao, recall, acuracia))
