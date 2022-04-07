import numpy as np

class AliasSampler:
    """
    alias table for arange topics.
    p: store probability np.ndarray,  [0.1, 0.5, 0.4]
    """
    def __init__(self, p, topics=None):
        self.topics = topics
        self.build_table(p, topics)

    def sample(self):
        u, k = np.modf(np.random.rand() * self._K) # modf函数可以返回输入的数字/向量的整数部分和小数部分
        # 所以相当于生成了0～K(exclude)的随机数  u是小数部分 k是整数部分
        # k决定了选那个竖条，u决定了选上半部分还是下半部分
        k = int(k)
        if u < self.v[k]:
            return k if self.topics is None else self.topics[k]
        else:
            return self.a[k] if self.topics is None else self.topics[self.a[k]]

    def build_table(self, p, topics):
        # p是采样概率分布
        self._K = len(p)  # K是分布的取值个数
        de = sum(p)       # 分母
        p = [each / de for each in p]  # 归一化得到分布  如果之前本来就是分布，也没有影响
        self.a = [0] * self._K  # a记录的是每一个短的是由那个长的补上的
        self.v = [each * self._K for each in p]   # 概率值 * K  相当于均值归一化 这样1割补分界线
        long, short = [], []
        for k, vk in enumerate(self.v):
            if 1. <= vk:
                long.append(k)
            else:
                short.append(k)

        while len(long) > 0 and len(short) > 0:
            l = long.pop()
            s = short.pop()
            self.a[s] = l
            self.v[l] -= (1. - self.v[s])  # 长的拿出一块补短 所以扣除
            if 1. > self.v[l]:
                short.append(l)
            else:
                long.append(l)
