import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from ldamodel import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# preprocessing
model = LDAModel().load_model('gibbs_no_7.5_0.1_neg3000500_50', 120)
model = LDAModel().load_model('gibbs_kRRp_tw_wo_7.5_0.1_neg3000500_50', 120)
# model = LDAModel().load_model('gibbs_kRRp_tw_wo_5.0_0.1_neg3000500_50', 200)



# model = LDAModel().load_model('gibbs_no_7.5_0.1_email3000500_30', 200)
# model = LDAModel().load_model('gibbs_kRRp_tw_wo_7.5_0.1_email3000500_30', 50)
# model = LDAModel().load_model('gibbs_kRRp_tw_wo_5.0_0.1_email3000500_30', 50)

ndk = model.get_ndk()[0:3000]

X = preprocessing.scale(ndk)

y = [1] * 1500 + [0] * 1500
# train
# clf = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=90)
clf = LogisticRegression(C=1.0, penalty='l2', tol=0.01, fit_intercept=True)
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

