#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from ldamodel import LDAModel
from evaluation import EvalTools
from preprocess import PreProcess
from dictionary import Dictionary
from protocol import Protocol
import getopt
import sys

K = 30
ITER_TIMES = 50
SAVE_FREQ = 5
MAX_DOC_NUM = 3000
MAX_WORD = 500
DIC_DOC_NAME = "dic_doc_spam_" + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
DIC_DOC_PATH = "./dic_docs/"
CORPUS_PATH = "./corpus/spam.csv"
IS_COMMAND = 1
IS_BUILD = 0
IS_TRAIN = 0
IS_DRAW = 1
RATIO = 1
SP = 1.0

# head + sample + '_' + prot_name + '_' + str(EPSILON) + str(DELTA) + '_b'
head = ""
sample = "gibbs"  # gibbs light
prot_name = "kRRp_tw_wo"  # kRR_wo, kRR_trunc, icde19_wo, kRRp_tw_wo, kRRp_tw_wnnt
EPSILON = 5.0  # 10 7.5 5 2.5
DELTA = 0.1
name = ""

if not IS_COMMAND:
    print("warning: your cmd line may not work as you supposed!!")

if IS_COMMAND:
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'e:d:t:s:p:m:r:k:x:O:h', ['eps=', 'delta=', 'itertimes=', 'sample=', 'protocol=', 'models=', 'resource=', 'n_topic=', 'ratio=', 'option=', 'help'])
    for opt_name, opt_val in opts:
        if opt_name == "-e" or opt_name == "--eps":
            eps = float(opt_val)
            EPSILON = eps
        elif opt_name == '--itertimes' or opt_name == "-t":
            ITER_TIMES = int(opt_val)
        elif opt_name == "-s" or opt_name == '--sample':
            sample = opt_val
        elif opt_name == "-p" or opt_name == "--protocol":
            prot_name = opt_val
        elif opt_name == "-O" or opt_name == "--option":
            if opt_val == 'B':
                IS_BUILD = True
            elif opt_val == 'T':
                IS_TRAIN = True
            elif opt_val == 'D':
                IS_DRAW = True
        elif opt_name == "-m" or opt_name == "--models":
            model_names = eval(opt_val)
            print(model_names)
        elif opt_name == "-d" or opt_name == "--delta":
            DELTA = opt_val
        elif opt_name == "-k" or opt_name == "--n_topic":
            K = int(opt_val)
        elif opt_name == "-r" or opt_name == "--resource":
            if opt_val == 'b':
                DIC_DOC_NAME = 'dic_doc_blogs_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
            elif opt_val == 'n':
                print("##")
                DIC_DOC_NAME = 'dic_doc_news_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
            elif opt_val == 'c':
                MAX_DOC_NUM = 1000
                RATIO = 1
                DIC_DOC_NAME = 'dic_doc_comments_' + str(MAX_DOC_NUM) + '_'+str(MAX_WORD)
            elif opt_val == 'm':
                DIC_DOC_NAME = 'dic_doc_movies_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
            elif opt_val == 'p':
                DIC_DOC_NAME = 'dic_doc_pos_neg_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
            elif opt_val == 'e':
                MAX_DOC_NUM = 3000
                RATIO = 10
                DIC_DOC_NAME = 'dic_doc_email_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
            elif opt_val == 's':
                MAX_DOC_NUM = 5000
                RATIO = 1
                DIC_DOC_NAME = 'dic_doc_spam_' + str(MAX_DOC_NUM) + '_' + str(MAX_WORD)
        elif opt_name == "-x" or opt_name == "--ratio":
            SP = float(opt_val)
        elif opt_name == "-h" or opt_name == "--help":
            print("-O | --option: operation option B build dictionary, T train model, D, draw pictures")
            print("-e | --eps: set eps, default 2.197")
            print("-d | --delta: set delta default 0.01")
            print("-t | --itertimes: set iteration time, default 30")
            print("-s | --sample: choose sample: gibbs or light")
            print("-k | --n_topic: set topic num")
            print("-p | --protocol: choose protocol: \
                        no, kRR, kRR_wo, kRR_wnnt, kRRp_tw, kRRp_tw_wo, kRRp_tw_wnnt, icde19, icde19_wo, icde19_wnnt")
            print("-r | --resource: train data set")
            print("-m | --models: set model name")
            print("-x | -- ratio: set sample ratio")
            print("-h | --help: print this info")
            exit(0)

if IS_BUILD:
    # pre processing
    preUnit = PreProcess().load_train(CORPUS_PATH, max_doc_num=MAX_DOC_NUM, max_word=MAX_WORD)
    # build dictionary
    docs = Dictionary().build_dic(w=preUnit.getW(), dic_doc_name=DIC_DOC_NAME, dic_doc_path=DIC_DOC_PATH)
else:
    docs = Dictionary().load_dic_doc(dic_doc_path=DIC_DOC_PATH, dic_doc_name=DIC_DOC_NAME)

print("V:", docs.V)
if name == "":
        name = head + sample + '_' + prot_name + '_' + str(EPSILON) + '_' + str(DELTA) + '_' +  DIC_DOC_NAME.split('_')[-3] \
        + DIC_DOC_NAME.split('_')[-2] + DIC_DOC_NAME.split('_')[-1] + '_' + str(K)
if SP < 0.9:
    name = name + '_' + str(SP)
print(name)

if IS_TRAIN:
    print("Run", name, sample, prot_name, str(EPSILON), str(DELTA), str(K), str(ITER_TIMES), str(SP), "...")
    prot = Protocol(eps=EPSILON, ratio=RATIO, delta=DELTA, protocol_type=prot_name, isps=True, V=docs.V)
    model = LDAModel(n_topic=K, model_name=name, docs=docs, maxL=100, spratio=SP)
    model.fit(protocol=prot, num_iterations=ITER_TIMES, method=sample, save_freq=SAVE_FREQ, batch_size=-1)

if IS_DRAW:
    model_names = [name]
    evalUnit = EvalTools(docs, [i for i in range(0, ITER_TIMES + 1, SAVE_FREQ)], [], model_names)
    for each in model_names:
        evalUnit.add_series(each)
    evalUnit.draw(x=[i for i in range(0, ITER_TIMES + 1, SAVE_FREQ)], file_name='a.pdf')
