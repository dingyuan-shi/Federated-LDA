import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from ldamodel import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# load
data = np.array(pd.read_csv("corpus/spam.csv", encoding="latin-1"))

# preprocessing
# model = LDAModel().load_model('gibbs_no_10.0_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_kRR_wo_7.5_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_kRR_trunc_7.5_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_icde19_wo_7.5_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_kRRp_tw_wo_7.5_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_kRRp_tw_wnnt_7.5_0.1_emails5000500_50', 50)
# model = LDAModel().load_model('gibbs_kRRp_tw_wnnt_5.0_0.1_emails5000500_50', 50)
model = LDAModel().load_model('gibbs_kRRp_tw_wo_5.0_0.1_emails5000500_50', 50)


ndk = model.get_ndk()
X = preprocessing.scale(ndk)
y = [0] * 2000 + [1] * 2000
# train
# clf = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(score)

# evaluation
print("======train======")
# 精准率
precision =  cross_val_score(clf,X_train,y_train,cv=5,scoring='precision')
print("平均精准率为: ",np.mean(precision))
# 召回率
recall =  cross_val_score(clf,X_train,y_train,cv=5,scoring='recall')
print("平均召回率为: ",np.mean(recall))
# F1值
f1 =  cross_val_score(clf,X_train,y_train,cv=5,scoring='f1')
print("平均F1值为: ",np.mean(f1))

print("======test======")
precision =  cross_val_score(clf, X_test, y_test, cv=5, scoring='precision')
print("平均精准率为: ", np.mean(precision))
# 召回率
recall =  cross_val_score(clf, X_test, y_test, cv=5, scoring='recall')
print("平均召回率为: ", np.mean(recall))
# F1值
f1 = cross_val_score(clf, X_test, y_test, cv=5, scoring='f1')
print("平均F1值为: ", np.mean(f1))
auc_score = roc_auc_score(y_test, y_pred)
print("AUC", auc_score)

