import numpy as np
import math
import random
import copy
from dictionary import Dictionary
from alias import AliasSampler
import pickle
import os
from protocol import Protocol


class LDAModel(object):

    def __init__(self, docs=None, n_topic=30, model_name="", num_mh=2, maxL=400, spratio=1.0) -> None:
        if docs is None:
            return
        self.K = n_topic
        self.perplexity = -1
        self._documents = docs.get_documents()
        self._V = docs.get_num_vocab()  # V单词总数
        self._D = docs.get_num_docs()    # D文档总数
        self._beta = 0.01
        self._Vbeta = self._V * self._beta
        self.denominator_part_beta_nk_or_beta = self.K * self._Vbeta
        self._alpha = 50 / self.K + 1
        self._sum_alpha = 0.1 * self.K
        self._nkv = [[0 for i in range(self._V)] for j in range(self.K)]
        self._ndk = [[0 for i in range(self.K)] for j in range(self._D)]
        self._nk = [0 for i in range(self.K)]
        self._z = []
        self.num_MH = num_mh
        self.model_name = model_name
        self._pri = docs.get_pri()
        self.maxL = maxL
        self.spratio=spratio

    def get_nkv(self):
        return self._nkv

    def get_ndk(self):
        return self._ndk

    def fit(self, protocol: Protocol, num_iterations=50, method='gibbs', model_name="", save_freq=5, batch_size=-1) -> None:
        np.random.seed(0)
        random.seed(0)
        if model_name != "":
            self.model_name = model_name
        for doc_id in range(self._D):
            doc_topic = [random.randint(0, self.K-1) for i in range(len(self._documents[doc_id]))]
            self._z.append(doc_topic)
            for word, topic in zip(self._documents[doc_id], doc_topic):
                self._nkv[topic][word] += 1
                self._ndk[doc_id][topic] += 1
                self._nk[topic] += 1
        if batch_size == -1:
            batch_size = self._D
        if method == 'gibbs':
            self._gibbs_sample(protocol, batch_size, num_iterations, save_freq)
        elif method == 'light':
            self._light_sample(protocol, batch_size, num_iterations, save_freq)
                
    def _gibbs_sample(self, protocol: Protocol, batch_size, n_interation=50, save_freq=5):
        for t in range(n_interation):
            if t % 10 == 0:
                print(t)
            if self.model_name != "" and t % save_freq == 0:
                self.save_model(self.model_name, t)
            batch = []
            B = [(sum(self._nkv[k]) + self._V * self._beta) for k in range(self.K)]
            #  for each doc
            for i in range(self._D):
                # doc_id, ndk, nkv, alpha, beta, B = paras
                paras = [i, self._ndk, self._nkv, self._alpha, self._beta, B]
                tuples = self._sample_per_doc_gibbs(i, self._alpha, self._beta, B, self.K)
                if len(tuples) == 0:
                    continue
                # rtuples = copy.copy(tuples)
                protocol.padding_and_sample(tuples, self.maxL, self.spratio)
                tuples_perturbed = protocol.perturb(tuples, paras)
                batch.append(tuples_perturbed)
                # 攒够了一批
                if batch_size == len(batch):
                    B = [(sum(self._nkv[k]) + self._V * self._beta) for k in range(self.K)]
                    protocol.accumulate(batch, i - batch_size + 1, self._nkv)
                    batch.clear()
            # 批大小不整除
            if len(batch) > 0:
                protocol.accumulate(batch, self._D - len(batch), self._nkv)
                batch.clear()

        if self.model_name != "" and n_interation % save_freq == 0:
            self.save_model(self.model_name, n_interation)
                    
    def _sample_per_doc_gibbs(self, i, alpha, beta, B, K):
        delta = []
        for j in range(len(self._documents[i])):
            word = self._documents[i][j]
            topic = self._z[i][j]
            self._ndk[i][topic] -= 1
            delta.append([0, word, topic, -1])
            A = (sum(self._ndk[i]) + K * alpha)
            # caculate probability
            p = [0 for i in range(K)]
            for k in range(K):
                # p[k] = (ndk[i][k] + alpha)*(nkw[k][word]+beta)/(beta*W + sum(nkw[k]))
                p[k] = (self._nkv[k][word] + beta) / B[k] * (self._ndk[i][k] + alpha) / A

            # sample
            for x in range(1, K):
                p[x] += p[x-1]
            samp = random.uniform(0, p[K-1])
            for x in range(K):
                if samp < p[x]:
                    topic = x
                    break
            delta.append([0, word, topic, 1])
            self._z[i][j] = topic
            self._ndk[i][topic] += 1
        return delta
        
    def _light_sample(self, protocol: Protocol, batch_size, num_iterations=50, save_freq=5):
        for ite in range(0, num_iterations):
            print(ite)
            denominator_nk_or_beta = sum(self._nk) + self.denominator_part_beta_nk_or_beta
            word_proposal_denom = [(self._nk[x] + self._Vbeta) for x in range(self.K)]
            beta_table = AliasSampler(p=[self._beta / word_proposal_denom[x] for x in range(len(word_proposal_denom))])
            word_tables = []
            for v in range(self._V):
                tmp = [self._nkv[x][v] for x in range(self.K)]
                topics = np.nonzero(tmp)[0]
                p = np.array([self._nkv[k][v] / word_proposal_denom[k] for k in topics])
                word_tables.append(AliasSampler(p=p, topics=topics))
        # 每次迭代
            if self.model_name != "" and ite % save_freq == 0:
                self.save_model(self.model_name, ite)
            batch = []
            # 对每个文档
            for d in range(self._D):
                # doc_id, ndk, nkv, alpha, beta, B = paras
                paras = [d, self._ndk, self._nkv, self._alpha, self._beta, None]
                tuples = self._sample_per_doc_light(d, denominator_nk_or_beta, word_tables, beta_table)
                if len(tuples) == 0:
                    continue
                # protocol.padding_and_sample(tuples, self.maxL, self.spratio)
                tuples_perturbed = protocol.perturb(tuples, paras)
                batch.append(tuples_perturbed)
                if len(batch) == batch_size:
                    protocol.accumulate(batch, d - batch_size + 1, self._nkv, self._nk)
                    denominator_nk_or_beta = sum(self._nk) + self.denominator_part_beta_nk_or_beta
                    word_proposal_denom = [self._nk[x] + self._Vbeta for x in range(self.K)]  # 常数提前算好
                    beta_table = AliasSampler(
                        p=[self._beta / word_proposal_denom[x] for x in range(len(word_proposal_denom))])
                    word_tables.clear()
                    for v in range(self._V):
                        tmp = [self._nkv[x][v] for x in range(self.K)]
                        topics = np.nonzero(tmp)[0]
                        p = np.array([self._nkv[k][v] / word_proposal_denom[k] for k in topics])
                        word_tables.append(AliasSampler(p=p, topics=topics))
                    batch.clear()
            # 批大小不整除
            if len(batch) > 0:
                protocol.accumulate(batch, self._D - len(batch), self._nkv, self._nk)
                batch.clear()

        if self.model_name != "" and num_iterations % save_freq == 0:
            self.save_model(self.model_name, num_iterations)
                
    def _sample_per_doc_light(self, d, denominator_nk_or_beta, word_tables, beta_table):
        # rndk = copy.copy(self._ndk)
        delta = []
        w_d = self._documents[d]
        N_d = len(w_d)
        # 对每个文档的每个单词
        for i, w in enumerate(w_d):
            old_topic = s = self._z[d][i]
            for _ in range(2):
                # word proposal
                nk_or_beta = np.random.rand() * denominator_nk_or_beta
                # print(denominator_nk_or_beta, denominator_part_beta_nk_or_beta)
                if nk_or_beta < self.denominator_part_beta_nk_or_beta:
                    t = beta_table.sample()
                else:
                    try:
                        t = word_tables[w].sample()
                    except:
                        t = random.randint(0, self.K - 1)

                # 采样得到新的话题
                if t != s:
                    nsw = self._nkv[s][w]
                    ntw = self._nkv[t][w]
                    ns = self._nk[s]
                    nt = self._nk[t]
                    
                    nsd_alpha = self._ndk[d][s] + self._alpha
                    ntd_alpha = self._ndk[d][t] + self._alpha
                    nsw_beta = nsw + self._beta
                    ntw_beta = ntw + self._beta
                    ns_Vbeta = ns + self._Vbeta
                    nt_Vbeta = nt + self._Vbeta
                    
                    proposal_nominator = nsw_beta * nt_Vbeta
                    proposal_denominator = ntw_beta * ns_Vbeta

                    if s == old_topic:
                        nsd_alpha -= 1
                        nsw_beta -= 1
                        ns_Vbeta -= 1

                    if t == old_topic:
                        ntd_alpha -= 1
                        ntw_beta -= 1
                        nt_Vbeta -= 1

                    pi_nominator = ntd_alpha * ntw_beta * ns_Vbeta * proposal_nominator
                    pi_denominator = nsd_alpha * nsw_beta * nt_Vbeta * proposal_denominator

                    if pi_denominator == 0:
                        continue
                    pi = pi_nominator / pi_denominator
                    m = -(np.random.rand() < pi)
                    s = (t & m) | (s & ~m)

                # doc proposal
                nd_or_alpha = np.random.rand() * (N_d + self._sum_alpha)
                if N_d > nd_or_alpha:
                    t = self._z[d][int(nd_or_alpha)]
                else:
                    t = random.randint(0, self.K - 1)

                if t != s:
                    nsd = self._ndk[d][s]
                    ntd = self._ndk[d][t]

                    nsd_alpha = proposal_nominator = nsd + self._alpha
                    ntd_alpha = proposal_denominator = ntd + self._alpha
                    nsw_beta = self._nkv[s][w] + self._beta
                    ntw_beta = self._nkv[t][w] + self._beta
                    ns_Vbeta = self._nk[s] + self._Vbeta
                    nt_Vbeta = self._nk[t] + self._Vbeta

                    if s == old_topic:
                        nsd_alpha -= 1
                        nsw_beta -= 1
                        ns_Vbeta -= 1

                    if t == old_topic:
                        ntd_alpha -= 1
                        ntw_beta -= 1
                        nt_Vbeta -= 1

                    pi_nominator = ntd_alpha * ntw_beta * ns_Vbeta * proposal_nominator
                    pi_denominator = nsd_alpha * nsw_beta * nt_Vbeta * proposal_denominator
                    # print(nsd_alpha, nsw_beta, nt_Vbeta, proposal_denominator)
                    if pi_denominator == 0:
                        continue
                    pi = pi_nominator / pi_denominator
                    m = -(np.random.rand() < pi)
                    s = (t & m) | (s & ~m)  # m=0没有更新 拒绝采样 m=1更新

            # update topic
            if s != old_topic:
                self._z[d][i] = s
                self._ndk[d][old_topic] -= 1
                self._ndk[d][s] += 1
                delta.append([0, w, old_topic, -1])
                delta.append([0, w, s, 1])
        return delta
                
    def save_model(self, file_name, n_iter):
        dir_name = os.path.join("./models/", file_name + "_" + str(n_iter))
        if os.path.exists(dir_name):
            os.remove(dir_name)
        with open(dir_name, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file_name, n_iter):
        with open(os.path.join("./models/", file_name + "_" + str(n_iter)), "rb") as f:
            self = pickle.load(f)
        return self

    def get_perplexity(self, n_iter):
        if self.perplexity == -1:
            e = 2.71828
            test_docs = self._documents
            phi = self._cal_phi(self._nkv, self._beta)
            # perplexity = exp-{sum(log(sum(p(z|d)*p(w|z)))/sum(N)}
            total_word = sum([len(test_docs[i]) for i in range(len(test_docs))])
            sum_log = 0
            for i in range(self._D):
                A = sum(self._ndk[i])
                if A == 0:
                    A = 1
                pzd = [self._ndk[i][x] / A for x in range(self.K)]
                for j in range(len(test_docs[i])):
                    pw = 0
                    word = test_docs[i][j]
                    for k in range(self.K):
                        pw += pzd[k] * phi[k][word]
                    if pw > 0.00001:
                        sum_log += math.log(pw)
            self.perplexity = pow(e, -sum_log / total_word)
            self.save_model(file_name=self.model_name, n_iter=n_iter)
        return self.perplexity

    def _cal_phi(self, nkw, beta):
        W = len(nkw[0])
        K = len(nkw)
        Phi = [[0 for i in range(W)] for j in range(K)]
        for k in range(K):
            Bk = (beta * W + sum(nkw[k]))
            for w in range(W):
                Phi[k][w] = (nkw[k][w] + beta) / Bk
        return Phi
