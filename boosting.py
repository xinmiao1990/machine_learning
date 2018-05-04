import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        N = len(features)
        y = np.array([0.0]*N)
        t = 0
        for h_t in self.clfs_picked:
            y += self.betas[t] * np.array(h_t.predict(features))
            t += 1
        result = np.array([-1]*N)
        result[y >= 0] = 1
        return np.array(result).tolist()


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # self.clfs, self.num_clf, self.T  are all given when call "boosting.AdaBoost(h_set, T=T)"
        # self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        # self.betas = []  # list of weights beta_t for t=0,...,T-1
        # t = 0
        # for ht in self.clfs:
        #     self.clfs_picked.append(ht)
        #     self.betas.append(0.5)
        #     if t >= self.T:
        #         break
        #     else:
        #         t += 1
        X = np.array(features)
        y = np.array(labels)
        N, D = X.shape
        # initialization
        w_t = np.array([1/N]*N)
        for t in range(0, self.T):
            # one pass through H set, find the one gives minimum error, name it "h_t"
            cnt = 0
            for h in self.clfs:
                hp = np.array(h.predict(features))
                met = np.sum(w_t[hp != y])
                if cnt == 0:
                    met_min = met
                    h_t = h
                elif met < met_min:
                    met_min = met
                    h_t = h
                cnt += 1

            hp = np.array(h_t.predict(features))
            eps_t = np.sum(w_t[hp != y])
            beta_t = 1/2*np.log((1-eps_t)/eps_t)
            self.betas.append(beta_t)
            self.clfs_picked.append(h_t)
            # update w_t
            w_tp1 = w_t * np.exp(beta_t)
            w_tp1[hp == y] = w_t[hp == y] * np.exp(-beta_t)
            w_tp1 = w_tp1/np.sum(w_tp1)
            w_t = w_tp1

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        X = np.array(features)
        y = np.array(labels)
        N, D = X.shape
        # initialization
        pi_t = np.array([0.5] * N)
        f_t = np.array([0.0] * N)
        for t in range(0, self.T):
            z_t = ((y+1)/2 - pi_t) / (pi_t * (1-pi_t))
            w_t = pi_t * (1-pi_t)
            # one pass through H set, find the one gives minimum error, name it "h_t"
            cnt = 0
            for h in self.clfs:
                hp = np.array(h.predict(features))
                met = np.sum(w_t * (z_t - hp)**2)
                if cnt == 0:
                    met_min = met
                    h_t = h
                elif met < met_min:
                    met_min = met
                    h_t = h
                cnt += 1

            hp = np.array(h_t.predict(features))
            f_t = f_t + 0.5 * hp
            self.betas.append(0.5)
            self.clfs_picked.append(h_t)
            # update pi_t
            pi_tp1 = 1 / (1 + np.exp(-2 * f_t))
            pi_t = pi_tp1

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
