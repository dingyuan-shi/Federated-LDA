import os
import re
import nltk
import random
import pandas as pd
import numpy as np

'''
CC 连词 and, or,but, if, while,although
CD 数词 twenty-four, fourth, 1991,14:24
DT 限定词 the, a, some, most,every, no
EX 存在量词 there, there's
FW 外来词 dolce, ersatz, esprit, quo,maitre
IN 介词连词 on, of,at, with,by,into, under
JJ 形容词 new,good, high, special, big, local
JJR 比较级词语 bleaker braver breezier briefer brighter brisker
JJS 最高级词语 calmest cheapest choicest classiest cleanest clearest
LS 标记 A A. B B. C C. D E F First G H I J K
MD 情态动词 can cannot could couldn't
NN 名词 year,home, costs, time, education
NNS 名词复数 undergraduates scotches
NNP 专有名词 Alison,Africa,April,Washington
NNPS 专有名词复数 Americans Americas Amharas Amityvilles
PDT 前限定词 all both half many
POS 所有格标记 ' 's
PRP 人称代词 hers herself him himself hisself
PRP$ 所有格 her his mine my our ours
RB 副词 occasionally unabatingly maddeningly
RBR 副词比较级 further gloomier grander
RBS 副词最高级 best biggest bluntest earliest
RP 虚词 aboard about across along apart
SYM 符号 % & ' '' ''. ) )
TO 词to to
UH 感叹词 Goodbye Goody Gosh Wow
VB 动词 ask assemble assess
VBD 动词过去式 dipped pleaded swiped
VBG 动词现在分词 telegraphing stirring focusing
VBN 动词过去分词 multihulled dilapidated aerosolized
VBP 动词现在式非第三人称时态 predominate wrap resort sue
VBZ 动词现在式第三人称时态 bases reconstructs marks
WDT Wh限定词 who,which,when,what,where,how
WP WH代词 that what whatever
WP$ WH代词所有格 whose
WRB WH副词
'''


class PreProcess:
    
    def __init__(self):
        self.w = None
        self.D = 0
        
        self.check_set = {'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNPS', 'PDT', 'RB',
            'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP'}
        self.illegal_set = {'<', '>', '|', '?', '!', ':', '@', '*', '(', ')', '%', '^', '%', '#', '$', '-', '+'}

    def load_train(self, data_path, max_word=600, max_doc_num=100):
        if data_path == './corpus/finefoods.txt':
            N, w_orig, word2cnt = self.load_comment(data_path, max_doc_num)
        elif data_path == "./corpus/movies/":
            N, w_orig, word2cnt = self.load_movies(data_path, max_doc_num, max_word)
        elif data_path == "./corpus/spam.csv":
            N, w_orig, word2cnt = self.load_spam(data_path, max_doc_num, max_word)
        elif data_path == "./corpus/emails/":
            N, w_orig, word2cnt = self.load_email(data_path, max_doc_num, max_word)
        elif data_path == "./corpus/pos_neg/":
            N, w_orig, word2cnt = self.load_pos_neg(data_path, max_doc_num, max_word)
        else:
            path = data_path
            fns = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
            # fns.sort()
            random.shuffle(fns)
            N, w_orig, word2cnt = self.load_blogs_news(fns, max_doc_num, max_word)
        w = []
        remove_list = set()
        for key in word2cnt:
            if word2cnt[key] <= 1:
                remove_list.add(key)
        for doc in w_orig:
            if len(doc) > max_word:
                doc = doc[0:max_word]
            doc_new = []
            for one_word in doc:
                if one_word not in remove_list:
                    doc_new.append(one_word)
            if len(doc_new) > max_word:
                doc_new = doc_new[0: max_word]
            w.append(doc_new)
        self.w = w
        self.D = N
        return self

    def load_pos_neg(self, data_path, N, W):
        path = os.path.join(data_path, 'pos/')
        pos = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
        new_pos = []
        for each in pos:
            if int(each.split('_')[-1].split('.')[0]) >= 9:
                new_pos.append(each)
                # print(each)
        new_pos.sort()
        path = os.path.join(data_path, 'neg/')
        neg = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
        new_neg = []
        for each in neg:
            if int(each.split('_')[-1].split('.')[0]) <= 2:
                new_neg.append(each)
                # print(each)
        new_neg.sort()
        w_orig = []
        cnt = 0
        for each in new_pos:
            text = open(each, encoding='ISO-8859-15').read()
            comment = self.preprocess_for_email(text)
            new_comment = []
            for word in comment:
                if len(word) > 3:
                    new_comment.append(word)
            if len(new_comment) > 100:
                w_orig.append(new_comment)
                cnt += 1
            if cnt == N // 2:
                break
        print(len(w_orig))
        N += len(w_orig)
        cnt = 0
        for each in new_neg:
            text = open(each, encoding='ISO-8859-15').read()
            comment = self.preprocess_for_email(text)
            new_comment = []
            for word in comment:
                if len(word) > 3:
                    new_comment.append(word)
            if len(new_comment) > 100:
                w_orig.append(new_comment)
                cnt += 1
                if cnt == N:
                    break
        print(len(w_orig))
        N = len(w_orig)
        word2cnt = dict()
        for new_comment in w_orig:
            print(len(new_comment), new_comment)
            for word in new_comment:
                if word not in word2cnt:
                    word2cnt[word] = 1
                else:
                    word2cnt[word] += 1
        return N, w_orig, word2cnt

    def load_email(self, data_path, N, W):
        path = os.path.join(data_path, 'ham/')
        hams = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
        hams.sort()
        path = os.path.join(data_path, 'spam/')
        spams = [os.path.join(root, fn) for root, dirs, files in os.walk(path) for fn in files]
        spams.sort()
        hams = hams[0: N // 2]
        spams = spams[0: N // 2]
        hams.extend(spams)
        w_orig = []
        word2cnt = dict()
        for each in hams:
            text = open(each, encoding='ISO-8859-15').read()
            email = self.preprocess_for_email(text)
            email = email[1:]
            new_email = []
            for word in email:
                if len(word) > 3:
                    new_email.append(word)
            w_orig.append(new_email)
            # print(len(new_email), new_email)
            for word in new_email:
                if word not in word2cnt:
                    word2cnt[word] = 1
                else:
                    word2cnt[word] += 1
        return N, w_orig, word2cnt

    def load_spam(self, path, N, W):
        data = np.array(pd.read_csv(os.path.join(path), encoding="latin-1").loc[:])
        N = min(len(data), N)
        w_orig = []
        word2cnt = dict()
        for i in range(N):
            email = self.preprocess_for_email(str(data[i][1]))
            print(email)
            if i % 10 == 0:
                w_orig.append(email)
            else:
                w_orig[-1].extend(email)
            for word in email:
                if word not in word2cnt:
                    word2cnt[word] = 1
                else:
                    word2cnt[word] += 1
        return N, w_orig, word2cnt

    def preprocess_for_email(self, text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        doc = []
        for word, pos in pos_tags:
            if pos in self.check_set and len(word) < 25 and word not in self.illegal_set:
                doc.append(word)
        return doc

    def load_movies(self, path, N, W):
        data = np.array(pd.read_csv(os.path.join(path, 'ratings.csv')).loc[:], dtype='int32')
        # N, w_orig, word2cnt
        if N >= data[-1][0]:
            N = data[-1][0]
        user2movies = dict()
        # for each user get movies
        cur_user = 0
        for i in range(len(data)):
            if data[i][0] - 1 != cur_user:
                cur_user += 1
                if cur_user == N:
                    break
            if cur_user not in user2movies:
                user2movies[cur_user] = []
            user2movies[cur_user].append(data[i][1])
        # for each movie: turn to its tag
        data = np.array(pd.read_csv(os.path.join(path, 'tags.csv')).loc[:])
        movie2tag = dict()
        for i in range(len(data)):
            if data[i][1] not in movie2tag:
                movie2tag[data[i][1]] = []
            movie2tag[data[i][1]].append(str(data[i][2]))
        # for each user turn his/her movie to tags
        w_orig = []
        for user in user2movies:
            tags_of_one_user = []
            for each_movie in user2movies[user]:
                if each_movie in movie2tag:
                    tags_of_one_user.extend(movie2tag[each_movie])
            w_orig.append(tags_of_one_user)
        for i in range(len(w_orig)):
            w_orig[i] = random.sample(w_orig[i], min(4 * W, len(w_orig[i])))
        word2cnt = dict()
        for doc in w_orig:
            for word in doc:
                if word not in word2cnt:
                    word2cnt[word] = 1
                else:
                    word2cnt[word] += 1
        return N, w_orig, word2cnt

    def load_blogs_news(self, files, N, maxword):
        if N > len(files):
            N = len(files)
        w_orig = []
        word2cnt = dict()
        for i in range(N):
            doc = self.preprocess_of_one_doc(files[i], maxword)
            for word in doc:
                if word in word2cnt:
                    word2cnt[word] += 1
                else:
                    word2cnt[word] = 1
            w_orig.append(doc)
        return N, w_orig, word2cnt
        
    def load_comment(self, data_path, max_doc_num):
        N = max_doc_num
        cnt = 0
        w_orig = []
        word2cnt = dict()
        for line in open(data_path, encoding='ISO-8859-1'):
            if line == "\n":
                continue
            elif line[0:12] == "review/text:":
                cnt += 1
                # print(cnt)
                doc = self.preprocess_for_comment(line[13:])
                for word in doc:
                    if word in word2cnt:
                        word2cnt[word] += 1
                    else:
                        word2cnt[word] = 1
                w_orig.append(doc)
            if cnt == N:
                break
        return N, w_orig, word2cnt

    def preprocess_of_one_doc(self, file_name, maxword):
        print("pre processing", file_name, "...")
        text = open(file_name, encoding='ISO-8859-15').read()
        # print(len(text), end="->")
        text = text[:min(len(text), maxword * 40)]
        # print(len(text))
        text = re.sub(u"<date>[^<]+</date>", u"", text)
        text = re.sub(u"<[^>]+>",u"",text)
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        doc = []
        for word, pos in pos_tags:
            if pos in self.check_set and len(word) < 25 and word not in self.illegal_set:
                doc.append(word)
        return doc
        
    def preprocess_for_comment(self, text):
        text = re.sub(u"<[^>]+>",u"",text)
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        doc = []
        for word, pos in pos_tags:
            if pos in self.check_set and len(word) < 25 and word not in self.illegal_set:
                doc.append(word)
        return doc
    
    def getW(self):
        return self.w
