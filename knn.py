# coding=utf-8


import datetime
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from numpy import interp
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score

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
param_grid = {'n_neighbors':[5,10,15,20]}
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf = KNeighborsClassifier()

i = 1
for train_index_outer, test_index_outer in outer_cv.split(x, y):
    x_train_outer, x_test_outer = x.iloc[train_index_outer], x.iloc[test_index_outer]
    y_train_outer, y_test_outer = y.iloc[train_index_outer], y.iloc[test_index_outer]
    # 内层循环

    sfs = SequentialFeatureSelector(estimator=rf,
                                    cv=inner_cv, k_features=10, verbose=2, n_jobs=-2, scoring='roc_auc')
    sfs.fit(x_train_outer,y_train_outer)
    # 获取最优的特征子集
    print('selected features:', sfs.k_feature_idx_)
    print("CV score:",sfs.k_score_)
    print(sfs.subsets_)
    grid_search =GridSearchCV(estimator=rf,param_grid=param_grid,scoring='roc_auc',cv=inner_cv,n_jobs=-2)

    grid_search.fit(x_train_outer,y_train_outer)
    #     获取最佳特征子集


    best_params =grid_search.best_params_
    print('最佳参数:',best_params)
    best_features = sfs.transform(x_train_outer)
    # 训练模型并在内层验证集上计算指标
    rf.set_params(**best_params)
    rf.fit(best_features,y_train_outer)

    test_features=sfs.transform(x_test_outer)
    y_pred = rf.predict(test_features)
    y_pred_pro = rf.predict_proba(test_features)[:, 1]
    # 特征重要性

    # print('特征重要性：',rf.feature_importances_)

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
plt.title('Selected 10 items ROC curves of 10-fold cross-validation and mean')
plt.legend(loc="lower right")
# 绘制特征数量和准确率曲线
plt.show()
print('准确率列表为：',outer_acc_scores)
print('召回率列表为：',outer_recall_scores)
print('精确率列表为：',outer_precision_scores)
print('f1分数列表为：',outer_f1_scores)
print('AUC列表为：',auc_list)
end_time = datetime.datetime.now()
print(end_time)