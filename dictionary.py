import pickle
# data from http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
# http://www.dataguru.cn/article-13407-1.html
import os
import copy

class Dictionary(object):
    
    def __init__(self):
        self.docs = []
        self.i2w = []
        self.w2i = {}
        self.w2c = {}
        self.doc_lengths = None
        self.V = 0
        self.D = 0
        self.pri = None
        self.ppri = None
        
    def build_dic(self, w, dic_doc_path, dic_doc_name=""):
        cnt = 0
        self.D = len(w)
        doc_length_list = []
        totalWord = 0
        for doc in w:
            totalWord += len(doc)
            for i in range(len(doc)):
                if doc[i] not in self.w2i:
                    self.w2i[doc[i]] = cnt
                    self.i2w.append(doc[i])
                    cnt += 1
                doc[i] = self.w2i[doc[i]]
            self.docs.append(doc)
            doc_length_list.append(len(doc))
            
        self.V = cnt
        self.doc_lengths = doc_length_list
        
        self.pri = [0 for i in range(cnt)]
        for doc in w:
            for i in range(len(doc)):
                self.pri[doc[i]] += 1
        for i in range(len(self.pri)):
            self.pri[i] /= totalWord
        self.ppri = copy.deepcopy(self.pri)
        for i in range(1, cnt):
            self.ppri[i] += self.ppri[i-1]
        if dic_doc_name != "":
            self.save_dic_doc(dic_doc_path, dic_doc_name)
        return self
            
    def save_dic_doc(self, dic_doc_path, dic_doc_name):
        dir_name = os.path.join(dic_doc_path, dic_doc_name)
        print(dir_name)
        if os.path.exists(dir_name):
            os.remove(dir_name)
        with open(dir_name, 'wb') as f:
            pickle.dump(self, f)
        
    def load_dic_doc(self, dic_doc_path, dic_doc_name):
        print(os.path.join(dic_doc_path, dic_doc_name))
        with open(os.path.join(dic_doc_path, dic_doc_name), 'rb') as f:
            a = pickle.load(f)
            self = a
            return self
        
    def get_documents(self) -> list:
        return self.docs

    def get_document(self, doc_id: int) -> list:
        return self.docs[doc_id]

    def get_vocabulary(self) -> dict:
        return self.w2i

    def get_num_docs(self) -> int:
        return self.D

    def get_num_vocab(self) -> int:
        return self.V

    def get_word(self, word_id: int) -> int:
        return self.i2w[word_id]

    def get_ith_doc_len(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def get_doc_lengths(self) -> list:
        return self.doc_lengths

    def get_pri(self):
        return self.pri

    def get_ppri(self):
        return self.ppri
