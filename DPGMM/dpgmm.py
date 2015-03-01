#!/usr/bin/python
# -*- coding: utf-8 -*-
# original: http://nbviewer.ipython.org/github/breakbee/PyNote/blob/master/Implementation_of_DPGMM.ipynb
# require python 3.2

import sys
import math
import numpy as np
import random
import functools
import copy
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import chi2
from scipy.stats import gamma
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

LOG_POSTERIOR_MIN = -1000000.0

class DPGMM:
    def __init__(self):
        return

    # 対数基底分布(対数ガウスーウィシャート)
    def log_G0(self, mu, prec):
        val = self.log_normalprob(mu, self.mu0, np.linalg.inv(self.beta*prec))
        val += self.log_wishartprob(prec, self.nu, self.S)
        return val

    # クラスタリングの現状を描画
    def plot_now(self):
        self.plot_counter += 1
        plt.clf()

        # サンプルの描画
        color_iter = itertools.cycle(['r', 'b', 'g'])
        for c in range(len(self.mu)):
            cX = np.array(list(zip(
                *list(filter(lambda _X_s: _X_s[1]==c, self.X_s))))[0])
            plt.plot(cX[:,0], cX[:,1], '.', color=cm.hot(float(c)/len(self.mu)))
            plt.title("#clusters=" + str(self.number_of_clusters) + 
                    " iter=" + str(self.iter_counter) +
                    " log_posterior=" + str(self.log_posterior))

        # パラメタ推定結果の描画
        color_iter = itertools.cycle(['r', 'b', 'g'])
        plt.axes()
        ax = plt.gca()
        for i, (mu, cov, pi, color) in enumerate(zip(self.mu, self.cov, self.pi, color_iter)):
            v, w = np.linalg.eigh(cov)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi
            ell = mpl.patches.Ellipse(mu, v[0]+2, v[1]+2, 180 + angle, color=color)
            ell.set_alpha(0.8)
            ax.add_patch(ell)

        filepath = "./result_%03d" % self.plot_counter
        plt.savefig(filepath)
        sys.stdout.write("\nsaved " + filepath)
        sys.stdout.flush()

    # ギブスサンプリングによるDPGMM
    def fit(self, X, iter_num):
        N, d = X.shape

        # CRP（クラスタリング事前分布）におけるハイパーパラメータ
        #   alpha:集中度パラメータ大きい値を設定すると、テーブル数が増えやすくなる．
        self.alpha = 1/(gamma.pdf(1,1))

        # 基底分布（ガウスーウィシャート分布）におけるハイパーパラメータ
        #   mu0:  クラスタ分布の平均パラメータの分布（ガウス分布）における平均パラメータ
        #   beta: クラスタ分布の平均パラメータの分布（ガウス分布）における精度パラメータにかかるスカラ
        #   nu:   クラスタ分布の精度パラメータの分布（ウィシャート分布）における自由度パラメータ
        #   S:    クラスタ分布の精度パラメータの分布（ウィシャート分布）における共分散行列パラメータ
        npa_X = np.array(X)
        mu_X = np.array([np.mean(npa_X[:,0]), np.mean(npa_X[:,1])])
        sigma_X = np.cov(npa_X, rowvar=0)

        self.mu0 = mu_X
        self.beta = 1/(gamma.pdf(1,1/d)) + (d-1)
        self.nu = d
        self.S = np.linalg.inv(sigma_X)

        # cache
        alpha = self.alpha
        beta = self.beta
        nu = self.nu
        mu0 = self.mu0
        S = self.S

        # 初期化
        #   c:     クラスタ数
        #   c_num: クラスタ別サンプル数
        #   cov:   クラスタ別共分散行列
        #   mu:    クラスタ別平均ベクトル
        #   s:     クラスタ別所属クラスタ
        c = 1
        c_num = [N]
        cov = [np.linalg.inv(self.wishartrand(self.nu, self.S))]
        mu = [np.random.multivariate_normal(self.mu0, np.linalg.inv(cov[0] * self.beta))]
        s = np.zeros(N, dtype=np.int32)

        # ギブスサンプリングにおけるカウンタ変数
        self.plot_counter = 0
        self.iter_counter = 0
        self.number_of_clusters = 1
        self.log_posterior = LOG_POSTERIOR_MIN

        # ギブスサンプリング
        sys.stdout.write("Gibbs Sampling Start.")
        log_posterior_max = LOG_POSTERIOR_MIN
        opt_s = copy.deepcopy(s)
        for iter_n in range(iter_num):
            self.iter_counter = iter_n
            sys.stdout.write("\n" 
                    + "Iteration:" + str(iter_n) + "/" + str(iter_num)
                    + " #clusters:" + str(c)
                    + " log_posterior_max: " + str(log_posterior_max))
            sys.stdout.flush()

            for i in range(N):
                # パターンX[i]を現在の所属クラスタから除去
                c_num[s[i]] -= 1
                if c_num[s[i]] == 0:
                    c -= 1
                    del(c_num[s[i]])
                    del(mu[s[i]])
                    del(cov[s[i]])
                    # クラスタインデックスの更新
                    for j in range(N):
                        if s[j] > s[i]:
                            s[j] -= 1

                # 所属クラスタの更新
                # 1.クラスタの事後確率の計算
                # - A)既存クラスタへ所属の場合
                p_s = np.zeros(c)
                for j in range(c):
                    p_s[j] = mvnorm.pdf(X[i], mu[j], cov[j]) * c_num[j] / (N - 1 + alpha)
                # - B)新規クラスタへ所属の場合
                tmp = (X[i] - mu0)[:, np.newaxis]
                Sb = np.linalg.inv(np.linalg.inv(S) + beta/(1+beta) * tmp * tmp.T)
                p_new = math.pow(beta/((1+beta)*math.pi), d/2)
                p_new *= math.pow(np.linalg.det(Sb), (nu+1)/2) * math.gamma((nu+1)/2)
                p_new /= math.pow(np.linalg.det(S), nu/2) * math.gamma((nu+1-d)/2)
                p_new *= alpha / (N - 1 + alpha)

                # 2.所属クラスタの決定
                sum_p_s = np.sum(p_s) + p_new
                rv = random.random()
                tmp = 0
                new_s = -1
                # - A)既存クラスタへ所属の場合
                for j in range(c):
                    tmp += p_s[j] / sum_p_s
                    if tmp > rv:
                        new_s = j
                        s[i] = j
                        c_num[j] += 1
                        break
                # - B)新規クラスタへ所属の場合
                if new_s == -1:
                    new_s = c
                    c += 1
                    s[i] = new_s
                    c_num.append(1)
                    # 事前分布(基底測度)からサンプリング
                    #mu.append(X[i])
                    #cov.append(np.eye(d))
                    new_prec = self.wishartrand(self.nu, self.S)
                    new_mu = np.random.multivariate_normal(
                            self.mu0, np.linalg.inv(new_prec * self.beta))
                    mu.append(new_mu)
                    cov.append(np.linalg.inv(new_prec))

            # 各クラスタパラメータの更新
            for j in range(c):
                # x_bar
                X_k = []
                sum_x_j = np.zeros(d)
                for k in range(N):
                    if s[k] == j:
                        sum_x_j += X[k]
                        X_k.append(X[k])
                x_bar = sum_x_j / c_num[j]

                # Sq^-1
                Sq = np.zeros((d, d))
                for k in range(c_num[j]):
                    tmp = (X_k[k] - x_bar)[:, np.newaxis]
                    Sq += tmp * tmp.T
                tmp = (x_bar - mu0)[:, np.newaxis]
                Sq += np.linalg.inv(S) + c_num[j]*beta/(c_num[j]+beta) * tmp * tmp.T
                Sq_inv = np.linalg.inv(Sq)

                # nu_c
                nu_j = nu + c_num[j]

                # prec_c
                prec_j = self.wishartrand(nu_j, Sq_inv)
                prec_c = (c_num[j] + beta) * prec_j
                mu_c = (c_num[j] * x_bar + beta * mu0)/(c_num[j] + beta)
                
                # クラスタパラメータの更新
                mu[j] = mvnorm.rvs(mu_c, np.linalg.inv(prec_c))
                cov[j] = np.linalg.inv(prec_j)

            # 事後確率最大化
            # 1.対数事後確率
            log_prior = self.log_ewens_sampling(c_num)
            log_prior -= self.log_factorial(c)

            sxmc = list(zip(s,X,mu,cov))
            log_normal = lambda _sxmc: self.log_normalprob(_sxmc[1], _sxmc[2], _sxmc[3])
            log_likelihood = functools.reduce(lambda accum,_c: accum + \
                    self.log_G0(mu[_c], np.linalg.inv(cov[_c])) + \
                    functools.reduce(lambda _accum,__sxmc:_accum + log_normal(__sxmc),
                        list(filter(lambda _sxmc: _sxmc[0]==_c, sxmc)), 0.0),
                    range(c), 0.0)

            log_posterior = log_prior + log_likelihood

            # 2.最大事後確率更新，割り当てパラメータ更新
            if log_posterior > log_posterior_max:
                log_posterior_max = log_posterior
                self.log_posterior = log_posterior
                self.number_of_clusters = c
                self.cov = cov
                self.mu = mu
                self.pi = [n/N for n in c_num]
                self.X_s = list(zip(X, s))
                self.plot_now()

        # パラメータの出力
        sys.stdout.write("\n" + "Iteration : finished")
        sys.stdout.flush()
        return self.X_s

    # -----------------------------------------------------
    # static method
    # ウィシャート分布からのサンプリング
    def wishartrand(self, nu, S):
        d = S.shape[0]
        chol = np.linalg.cholesky(S)
        tmp = np.zeros((d, d))
        for i in range(d):
            for j in range(i+1):
                if i==j:
                    tmp[i, j] = np.sqrt(chi2.rvs(nu-i))
                else:
                    tmp[i, j] = np.random.normal(0, 1)
        return np.dot(chol, np.dot(tmp, np.dot(tmp.T, chol.T)))

    # 対数ウィシャート分布
    def log_wishartprob(self, X, nu, S):
        d = X.shape[0]
        val = (nu-d-1)*0.5*math.log(np.linalg.det(X))
        val += -0.5 * np.trace(np.linalg.inv(S)*X)
        val -= (nu*d*0.5)*math.log(2)
        val -= (d*(d-1)*0.25)*math.log(math.pi)
        val -= (nu*0.5)*math.log(np.linalg.det(S))
        val -= functools.reduce(lambda accum,j: accum + math.log(math.gamma((nu+1-j)*0.5)), range(1,d+1), 0.0)
        return val

    # 対数ガウス分布
    def log_normalprob(self, X, mu, cov):
        d = mu.shape[0]
        diff_mat = np.mat(X-mu)
        prec_mat = np.mat(np.linalg.inv(cov))
        val = -0.5*float(diff_mat*prec_mat*diff_mat.T)
        val -= d*0.5*math.log(2*math.pi)
        val -= 0.5*math.log(np.linalg.det(cov))
        return val

    # 対数イーウェンスの抽出公式
    def log_ewens_sampling(self, c_num):
        n = functools.reduce(lambda a,b: a+b, c_num, 1)
        val = math.log(self.alpha) * len(c_num)
        val += functools.reduce(lambda accum,ni: accum + self.log_factorial(ni), c_num, 0.0)
        val -= self.log_ascending_factorial(self.alpha, n)
        return n

    # 対数階乗
    def log_factorial(self, n):
        return functools.reduce(lambda accum,e: accum + math.log(e), range(1,n+1), 0.0)

    # 対数上昇階乗
    def log_ascending_factorial(self, a, n):
        return functools.reduce(lambda accum,e: accum + math.log(a+e), range(n), 0.0)

