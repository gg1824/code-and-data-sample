# coding=utf-8
import datetime
import pandas as pd
import numpy as np
# 引入算法
import xgboost as xgb
from mlxtend.feature_selection import SequentialFeatureSelector as Sfs
from numpy import interp
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix

start_time = datetime.datetime.now()
print(start_time)
f = open(r'D:\python\workspace\全新\Imblearn上采样.csv')
data = pd.read_csv(f)
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
# 外层准确率列表
outer_acc_scores = []
# 外层precision列表
outer_precision_scores = []
# 外层f1分数列表
outer_f1_scores = []
# 外层召回率列表
outer_recall_scores = []

# 初始化存储每个分割的ROC曲线和AUC的列表
tpr_s = []
auc_list = []

all_auc = []
mean_fpr = np.linspace(0, 1, 100)
# param_grid = {'n_estimators':100,'max_depth':6,'learning_rate':0.1,'gamma':0,''
#
#
#               }


inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


i = 1
for train_index_outer, test_index_outer in outer_cv.split(x, y):
    x_train_outer, x_test_outer = x.iloc[train_index_outer], x.iloc[test_index_outer]
    y_train_outer, y_test_outer = y.iloc[train_index_outer], y.iloc[test_index_outer]
    # 内层循环
    estimator = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=50,max_depth=6, min_child_weight=3)

    # gamma 默认0  learning_rate =0.3   max_depth=6, min_child_weight=1  subsample=1
    selector = Sfs(estimator=estimator, k_features=10, cv=inner_cv, scoring='roc_auc', verbose=2,n_jobs=-2)
    selector.fit(x_train_outer, y_train_outer)
    # 获取最优的特征子集
    print('selected features:', selector.k_feature_idx_)
    print("CV score:", selector.k_score_)
    print(selector.subsets_)
    x_train_selected = selector.transform(x_train_outer)
    x_test_selected = selector.transform(x_test_outer)

    estimator.fit(x_train_selected, y_train_outer)


    y_pred = estimator.predict(x_test_selected)
    print('y_pred:', y_pred)
    cm=confusion_matrix(y_test_outer,y_pred)
    print(cm)


# y_test_outer对应的每个样本被预测为有问题的概率
    y_pred_pro = estimator.predict_proba(x_test_selected)[:, 1]
# print('被预测为有问题的概率是：',y_pred_pro)

    fpr, tpr, thresholds = roc_curve(y_test_outer, y_pred_pro, pos_label=1)
    print('阈值是：', thresholds)
    tpr_s.append(interp(mean_fpr, fpr, tpr))
    tpr_s[-1][0] = 0.0

    # 非最佳阈值下的指标
    # 准确率
    acc = accuracy_score(y_test_outer, y_pred)
    outer_acc_scores.append(acc)
    print('accuracy：', acc)
    # 精确率
    precision = metrics.precision_score(y_test_outer, y_pred)
    outer_precision_scores.append(precision)
    print('precision：', precision)
    # AUC分数
    auc_score = roc_auc_score(y_test_outer, y_pred_pro)
    auc_list.append(auc_score)
    print('AUC：', auc_score)
    # 召回率
    recall = metrics.recall_score(y_test_outer, y_pred)
    outer_recall_scores.append(recall)
    print('召回率：', recall)
    # f1分数
    f1 = f1_score(y_test_outer,y_pred)
    outer_f1_scores.append(f1)
    print('f1:',f1)
    max_index = (tpr - fpr).tolist().index(max(tpr - fpr))
    max_threshold = thresholds[max_index]
    print('最好阈值是：', max_threshold)
    plt.plot(fpr, tpr, label='ROC curve fold%d (AUC=%0.2f)' % (i, auc_score))
    i += 1
# 画对角线，画平均ROC曲线
plt.plot([0, 1], [0, 1], color='r', linestyle="--", label='Random Guessing')
mean_tpr = np.mean(tpr_s, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC=%0.2f)' % mean_auc)
plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Selected 1items ROC curves of 10-fold cross-validation and mean')
plt.legend(loc="lower right")
# 绘制特征数量和准确率曲线
plt.show()
print('准确率列表为：',outer_acc_scores)
print('召回率列表为：',outer_recall_scores)
print('精确率列表为：',outer_precision_scores)
print('f1分数列表为：',outer_f1_scores)
print('AUC列表为：', auc_list)
end_time = datetime.datetime.now()
print(end_time)