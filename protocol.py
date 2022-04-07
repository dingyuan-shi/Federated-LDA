import random
import copy
import numpy as np
import math


class Protocol:

    def __init__(self, eps=1.1, delta=0, ratio = 1, protocol_type="no", isps=False, V=2500):
        self.protocol_type = protocol_type
        self.epsilon = eps
        if len(protocol_type) > 4 and protocol_type[3] == 'p':
            self.prob = 1 - 1 / (0.01 * math.pow(2.71828, eps) + 1)
        else:
            self.prob = 1 - 1 / (1 * ratio / V * math.pow(2.71828, eps) + 1)
        print("1 - eta = ", self.prob) 
        self.isps = isps
        self.delta = float(delta)
        self.for_delta = []
        self.for_delta_pref = []
        self.for_delta_tag = False

    def _update(self, nkv):
        self.for_delta = copy.deepcopy(nkv)
        for i in range(len(self.for_delta)):
            self.for_delta[i].sort()
        K = len(nkv)
        W = len(nkv[0])
        self.for_delta_pref = []
        for k in range(K):
            pref = copy.deepcopy(self.for_delta[k])
            for w in range(1, W):
                pref[w] += pref[w - 1]
            self.for_delta_pref.append(pref)
        self.for_word_sample = []
        coeff_k = [self.for_delta_pref[k][-1] for k in range(K)]
        for topic in range(K):
            p = [each / coeff_k[topic] for each in nkv[topic]]
            for t in range(1, W):
                p[t] += p[t - 1]
            self.for_word_sample.append(p)
    

    def perturb(self, tuples: list, paras=None):
        doc_id, ndk, nkv, alpha, beta, B = paras
        if not self.for_delta_tag:
            self.for_delta_tag = True
            self._update(nkv)
            print('should appear only once')
        K = len(nkv)
        W = len(nkv[0])
        if self.protocol_type == "no":
            return self._perturb_no(tuples)
        elif self.protocol_type == "kRR_trunc":
            return self._perturb_kRR(tuples, doc_id, ndk, nkv, K, W, True)
        elif self.protocol_type == "kRR_wo":
            return self._perturb_kRR(tuples, doc_id, ndk, nkv, K, W, False)
        elif self.protocol_type == "kRRp_tw_wo":
            return self._perturb_kRRp_tw(tuples, doc_id, ndk, nkv, K, W)
        elif self.protocol_type == "icde19_wo":
            return self._perturb_icde_wo(tuples, doc_id, ndk, nkv, K, W)
        else:
            print("ERROR: didn't apply any protocol with accident")
            return tuples

    def _in_delta(self, loc, topic, nkv):
        if self.delta == 0:
            return False
        rval = nkv[topic][loc] 
        k = len(nkv)
        x = self._upper_bound(self.for_delta[topic], rval)
        r = random.random()
        is_in = (self.for_delta_pref[topic][x]) / self.for_delta_pref[topic][-1] <= self.delta and r < 2 * self.delta
        # print(is_in)
        return is_in


    def _perturb_no(self, tuples):
        return tuples


    def _perturb_kRR(self, tuples, doc_id, ndk, nkv, K, W, is_trunc):

        for i in range(0, len(tuples), 2):
            r = random.random()
            if r <= self.prob:
                continue
            else:
                word = tuples[i][1]
                if is_trunc and self._in_delta(word, tuples[i][2], nkv):
                    continue
                word = random.randint(0, W - 1)
                tuples[i][1] = word
                tuples[i + 1][1] = word
        return tuples

    def _upper_bound(self, arr, target):
        i = 0
        j = len(arr)
        while i < j:
            mid = i + (j - i) // 2
            if target > arr[mid]:
                i = mid + 1
            else:
                j = mid
        return i
    
    def _lower_bound(self, arr, target):
        i = 0
        j = len(arr)
        while i < j:
            m = i + (j-i) // 2
            if arr[m] >= target:
                j = m
            else:
                i = m + 1
        return i


    def _perturb_kRRp_tw(self, tuples, doc_id, ndk, nkv, K, W):
        coeff = sum(ndk[doc_id])
        pri_topic = [each / coeff for each in ndk[doc_id]]
        for i in range(1, K):
            pri_topic[i] += pri_topic[i - 1]
        for i in range(0, len(tuples), 2):
            r = random.random()
            if r <= self.prob:
                continue
            else:
                samp = random.uniform(0, pri_topic[K - 1])
                topic = 0
                while samp > 0:
                    samp -= pri_topic[topic]
                    topic += 1
                if topic >= K:
                    topic = K - 1
                # sample new word via topic
                p = self.for_word_sample[topic]
                samp = random.uniform(0, p[W - 1])
                word = 0
                while samp > 0:
                    samp -= p[word]
                    word += 1
                if word >= W:
                    word = W - 1
                if self._in_delta(word, topic, nkv):
                    continue
                
                tuples[i][1] = word
                tuples[i + 1][1] = word
        return tuples

    def _perturb_icde_wo(self, tuples, doc_id, ndk, nkv, K, W):
        total_num = len(tuples) // 2
        if total_num == 0:
            return tuples
        cand_size = int((1-self.prob) * total_num)
        if cand_size > total_num:
            cand_size = total_num
        if cand_size == 0:
            cand_size = 1
        epsp = self.prob / cand_size 
        perturb_cand = random.sample([i for i in range(0, len(tuples), 2)], cand_size)
        for each in perturb_cand:
            word = tuples[each][1]
            r = random.random()
            if r > epsp:
                kp = random.randint(0, W-2)
                if kp < word:
                    word = kp
                else:
                    word = kp + 1
            tuples[each][1] = word
            tuples[each + 1][1] = word
        return tuples

    def accumulate(self, batch, startid, nkv, nk=None, ndk=None):
        if self.protocol_type == "no" or self.protocol_type == "kRR_wo":
            self._acc_naive(batch, startid, nkv, nk, ndk)
        elif self.protocol_type == "kRR_trunc":
            self._acc_naive_sort(batch, startid, nkv, nk, ndk)
        elif self.protocol_type == "kRRp_tw_wo":
            self._acc_naive_sort(batch, startid, nkv, nk, ndk)
        elif self.protocol_type == "icde19_wo":
            self._acc_icde(batch, startid, nkv, nk, ndk)
        else:
            print("ERROR: didn't apply any protocol with accident")
            pass

    def _acc_naive(self, batch, startid, nkv, nk=None, ndk=None):
        for j in range(len(batch)):
            each_delta = batch[j]
            for i in range(0, len(each_delta), 2):
                if nkv[each_delta[i][2]][each_delta[i][1]] - 1 >= 0:
                    nkv[each_delta[i][2]][each_delta[i][1]] -= 1
                    nkv[each_delta[i + 1][2]][each_delta[i + 1][1]] += 1
                    if ndk is not None and ndk[j + startid][each_delta[i][2]] - 1 >= 0:
                        ndk[j + startid][each_delta[i][2]] -= 1
                        ndk[j + startid][each_delta[i + 1][2]] += 1
        K = len(nkv)
        W = len(nkv[0])
        self.for_word_sample = []
        coeff_k = [sum(nkv[k]) for k in range(K)]
        for topic in range(K):
            p = [each / coeff_k[topic] for each in nkv[topic]]
            for t in range(1, W):
                p[t] += p[t - 1]
            self.for_word_sample.append(p)
        if nk:
            for k in range(len(nkv)):
                nk[k] = sum(nkv[k])

    def _acc_naive_sort(self, batch, startid, nkv, nk=None, ndk=None):
        for j in range(len(batch)):
            each_delta = batch[j]
            for i in range(0, len(each_delta), 2):
                if nkv[each_delta[i][2]][each_delta[i][1]] - 1 >= 0:
                    nkv[each_delta[i][2]][each_delta[i][1]] -= 1
                    nkv[each_delta[i + 1][2]][each_delta[i + 1][1]] += 1
                    if ndk is not None and ndk[j + startid][each_delta[i][2]] - 1 >= 0:
                        ndk[j + startid][each_delta[i][2]] -= 1
                        ndk[j + startid][each_delta[i + 1][2]] += 1

        self._update(nkv)
        if nk:
            for k in range(len(nkv)):
                nk[k] = sum(nkv[k])

    def _acc_icde(self, batch, startid, nkv, nk=None, ndk=None):
        for j in range(len(batch)):
            each_delta = batch[j]
            for i in range(0, len(each_delta), 2):
                if each_delta[i][1] == -1 or each_delta[i][2] == -1 or each_delta[i + 1][1] == -1 or each_delta[i + 1][2] == -1:
                    continue
                if nkv[each_delta[i][2]][each_delta[i][1]] - 1 >= 0:
                    nkv[each_delta[i][2]][each_delta[i][1]] -= 1
                    nkv[each_delta[i + 1][2]][each_delta[i + 1][1]] += 1
                    if ndk is not None and ndk[j + startid][each_delta[i][2]] - 1 >= 0:
                        ndk[j + startid][each_delta[i][2]] -= 1
                        ndk[j + startid][each_delta[i + 1][2]] += 1

        if nk:
            for k in range(len(nkv)):
                nk[k] = sum(nkv[k])

    def padding_and_sample(self, tuples, maxL=0, sample_size=0):
        if self.isps and sample_size > 0.1:
            self._padding(tuples, maxL)
            self._sample(tuples, sample_size)

    def _padding(self, tuples, maxL):
        orig_size = len(tuples)
        padding_size = maxL - orig_size
        while padding_size > 0:
            padding_size -= 2
            idx = random.randint(0, orig_size - 1)
            if idx % 2 == 0:
                tuples.append(copy.deepcopy(tuples[idx]))
                tuples.append(copy.deepcopy(tuples[idx + 1]))
            else:
                tuples.append(copy.deepcopy(tuples[idx - 1]))
                tuples.append(copy.deepcopy(tuples[idx]))

    def _sample(self, tuples, spratio):
        if spratio > 0.99:
            return
        sample_size = int(len(tuples)//2 * spratio)
        if sample_size > len(tuples):
            sample_size = len(tuples)
        new_tuples = []
        for i in range(sample_size):
            idx = random.randint(0, len(tuples) - 1)
            if idx % 2 == 0:
                new_tuples.append(tuples[idx])
                new_tuples.append(tuples[idx + 1])
            else:
                new_tuples.append(tuples[idx - 1])
                new_tuples.append(tuples[idx])
        # print(len(tuples), len(new_tuples))
        tuples = new_tuples
