from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
'''Read dataset1, LDA prediction'''
'''Read dataset1'''
def read_file1():

    train_id = np.loadtxt("../data/dataset1/mi_lnc_train_id1.txt")
    test_id = np.loadtxt("../data/dataset1/mi_lnc_test_id1.txt")
    #neg_id = np.loadtxt("data/MLI_negtive_id.txt")

    #low_A = np.loadtxt("dataset1_result/low_A_256.txt")
    #low_A = np.loadtxt("../dataset1_result/low_A_256_origin_data1.txt")
    low_A = np.loadtxt("../dataset1_result/low_A_256_ACL_data1.txt")
    mi_lnc = np.loadtxt("../data/dataset1/yuguoxian_lnc_mi.txt").T
    neg_id=[]
    mi_feature = low_A[645: ]
    lnc_feature = low_A[ :240]
    return train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, neg_id

'''Read  dataset2'''
def read_file2():

    train_id = np.loadtxt("../data/dataset2/mi_lnc_train_id1.txt")
    test_id = np.loadtxt("../data/dataset2/mi_lnc_test_id1.txt")
    #neg_id = np.loadtxt("../data/dataset2/MLI_negtive_id.txt")
    neg_id=[]
    #low_A = np.loadtxt("../dataset1_result/low_A_256_origin_data2.txt")
    low_A = np.loadtxt("../dataset1_result/low_A_256_ACL_data2.txt")

    mi_lnc = pd.read_csv('../data/dataset2/mi_lnc_intersection.csv', index_col='Unnamed: 0')
    mi_lnc = mi_lnc.values

    mi_feature = low_A[981: ]
    # print(mi_feature.shape)
    lnc_feature = low_A[:665]
    return train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, neg_id

def get_feature(A_feature, B_feature, index, adi_matrix):
    input = []
    output = []
    for i in range(index.shape[0]):
        A_i = int(index[i][0])
        B_j = int(index[i][1])
        feature = np.hstack((A_feature[A_i], B_feature[B_j]))
        input.append(feature.tolist())
        label = adi_matrix[[A_i],[B_j]].tolist()
        # print(type(label))
        # label = label.tolist()
        # print(label)
        output.append(label)
    output = np.array(output)
    output = output.ravel()
    return np.array(input), output

def aupr_f1(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = average_precision_score(y_true,y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    return aupr, f1


'''miRNA-lncRNA'''
#train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file1()
train_id, test_id, low_A, mi_lnc, mi_feature, lnc_feature, negtive_id = read_file2()
train_input, train_output = get_feature(mi_feature, lnc_feature, train_id, mi_lnc)  # (2328, dim)
test_input, test_output = get_feature(mi_feature, lnc_feature,test_id, mi_lnc)
#case_study_input,case_study_output = get_feature(mi_feature,lnc_feature,negtive_id,mi_lnc)


# --------------------------------- Exploring the performance of different classifiers------------------------------
'''AdaBoost'''
flag = 0
#flag = 1
if flag:
    ada = AdaBoostClassifier(n_estimators=40)
    ada.fit(train_input,train_output)
    y_pred = ada.predict_proba(test_input)[:,1]
    print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("ada auc:", auc, "aupr:", aupr, "F1:", f1)
'''XGBoost'''
#flag = 0
# flag = 1
if flag:
    xgb = XGBClassifier(n_estimators = 200, eta = 0.1, max_depth = 10)
    xgb.fit(train_input,train_output)
    y_pred = xgb.predict_proba(test_input)[:,1]
    print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("xgb auc:", auc, "aupr:", aupr, "F1:", f1)


'''LigheGBM'''
#flag = 0
#flag = 1
if flag:
    lgb_train = lgb.Dataset(train_input, train_output)
    lgb_eval = lgb.Dataset(test_input, test_output, reference=lgb_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'max_depth': 4,
        'metric': {'l2', 'auc'},
        'is_unbalance': 'true',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round = 100,
                    valid_sets = lgb_eval,
                    early_stopping_rounds = 300)
    y_pred = gbm.predict(test_input, num_iteration = gbm.best_iteration)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("lightauc:", auc, "aupr:", aupr, "F1:", f1)


'''RandomForest'''
# flag = 0
#flag = 1
if flag:

    rf = RandomForestClassifier(n_estimators = 500, max_depth = 7)
    rf.fit(train_input,train_output)
    y_pred = rf.predict_proba(test_input)[:,1]
    print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("RF auc:", auc, "aupr:", aupr, "F1:", f1)

'''MLP'''
flag = 0
#flag = 1
if flag:
    mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(512,2),activation='relu', learning_rate_init=0.0001,max_iter=1000)
    mlp.fit(train_input,train_output)
    y_pred = mlp.predict_proba(test_input)[:,1]
    #print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("MLP auc:", auc, "aupr:", aupr, "F1:", f1)

'''GBDT'''
#flag = 0
#flag = 1
if flag:

    gbdt = GradientBoostingClassifier(n_estimators=100,max_depth = 10)
    gbdt.fit(train_input, train_output)

    y_pred = gbdt.predict_proba(test_input)[:,1]
    #print(y_pred)
    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)
    print("GBDT auc:", auc, "aupr:", aupr, "F1:", f1)

'''MLP-torch'''
#flag = 0
flag = 1
if flag:
    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.long)
    test_input = torch.tensor(test_input, dtype=torch.float32)

    # 创建MLPClassifier模型
    input_size = train_input.shape[1]
    hidden_sizes = [512, 2]
    num_classes = 2
    mlp = MLP(input_size, hidden_sizes, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:3"
    mlp.to(device)
    train_input = train_input.to(device)
    train_output = train_output.to(device)
    test_input = test_input.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = mlp(train_input)
        loss = criterion(outputs, train_output)
        # loss=suploss(outputs,train_output)
        print(epoch, loss.item())
        loss.backward()
        optimizer.step()

    # 在测试集上进行预测
    with torch.no_grad():
        mlp.eval()
        y_pred = mlp(test_input)
        y_pred = y_pred[:, 1].cpu().numpy()

    auc = roc_auc_score(test_output, y_pred)
    aupr, f1 = aupr_f1(test_output, y_pred)

    print("MLP auc:", auc, "aupr:", aupr, "F1:", f1)