import numpy as np
from ldamodel import *
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

RS = 25


X = LDAModel().load_model('gibbs_no_7.5_0.1_neg3000500_50', 100).get_ndk()
X = sklearn.preprocessing.scale(X)[:3000, :]
y = [1] * 1500 + [0] * 1500
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=RS)

clf = SGDClassifier(random_state=RS)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# evaluation
pre = sklearn.metrics.accuracy_score(y_train, y_pred_train)
recall = sklearn.metrics.accuracy_score(y_train, y_pred_train)
f1 = sklearn.metrics.f1_score(y_train, y_pred_train)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_train)
# print("train: pre", pre, "recall", recall, "f1", f1, "auc", auc)

pre = sklearn.metrics.accuracy_score(y_test, y_pred)
recall = sklearn.metrics.recall_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
print("test: pre", pre, "recall", recall, "f1", f1, "auc", auc)

##################
X = LDAModel().load_model('gibbs_kRRp_tw_wo_7.5_0.1_neg3000500_50', 100).get_ndk()
# X = LDAModel().load_model('gibbs_kRRp_tw_wo_5.0_0.1_neg3000500_50', 80).get_ndk()
X = sklearn.preprocessing.scale(X)[:3000, :]
y = [1] * 1500 + [0] * 1500
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=RS)

clf = SGDClassifier(loss="log", penalty='l2', alpha=0.45, l1_ratio=0.05,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                  epsilon=0.1,
                 random_state=RS, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

pre = sklearn.metrics.accuracy_score(y_test, y_pred)
recall = sklearn.metrics.recall_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
print("test: pre", pre, "recall", recall, "f1", f1, "auc", auc)


##################
X = LDAModel().load_model('gibbs_kRRp_tw_wo_5.0_0.1_neg3000500_50', 100).get_ndk()
X = sklearn.preprocessing.scale(X)[:3000, :]
y = [1] * 1500 + [0] * 1500
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=RS)

# clf = LogisticRegression()
clf = SGDClassifier(loss="hinge", penalty='l2', alpha=0.45, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                  epsilon=0.1,
                 random_state=RS, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

pre = sklearn.metrics.accuracy_score(y_test, y_pred)
recall = sklearn.metrics.recall_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
print("test: pre", pre, "recall", recall, "f1", f1, "auc", auc)
