import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split,KFold


def create_train_test_sets_cv2(m, n,association_matrix, random_state=None):
    rna_ids = np.array([i for i in range(m)])
    dis_ids = np.array([i for i in range(n)])

    known_indices = np.argwhere(association_matrix == 1)

    negative_indices = np.argwhere(association_matrix == 0)


    selected_negative_indices = negative_indices[
        np.random.choice(len(negative_indices), len(known_indices), replace=False)]


    all_indices = np.concatenate((known_indices, selected_negative_indices))


    labels = np.concatenate((np.ones(len(known_indices)), np.zeros(len(selected_negative_indices))))


    rna_ids_all = rna_ids[all_indices[:, 0]]
    dis_ids_all = dis_ids[all_indices[:, 1]]


    train_rna_ids, test_rna_ids, train_dis_ids, test_dis_ids = train_test_split(
        rna_ids_all, dis_ids_all, test_size=0.2, random_state=random_state
    )


    train_set = np.column_stack((train_rna_ids, train_dis_ids))
    test_set = np.column_stack((test_rna_ids, test_dis_ids))

    return train_set, test_set
def create_train_test_sets(m, n, association_matrix, negative_ratio=1, random_state=None):

    rna_ids = np.array([i for i in range(m)])
    dis_ids = np.array([i for i in range(n)])

    positive_indices = np.random.permutation(np.argwhere(association_matrix > 0))

    negative_indices = np.argwhere(association_matrix == 0)


    positive_subsets = np.array_split(positive_indices, 5)

    test_positive_indices = positive_subsets[0]
    train_positive_indices = np.concatenate(positive_subsets[1:])

    num_train_negatives = int(negative_ratio * len(train_positive_indices))
    train_negative_indices = negative_indices[np.random.choice(len(negative_indices), num_train_negatives, replace=False)]

    train_df = pd.DataFrame(data={
        "ID1": np.concatenate((rna_ids[train_positive_indices[:, 0]], rna_ids[train_negative_indices[:, 0]])),
        "ID2": np.concatenate((dis_ids[train_positive_indices[:, 1]], dis_ids[train_negative_indices[:, 1]])),

    })

    test_df = pd.DataFrame(data={
        "ID1": np.concatenate((rna_ids[test_positive_indices[:, 0]], rna_ids[negative_indices[:, 0]])),
        "ID2": np.concatenate((dis_ids[test_positive_indices[:, 1]], dis_ids[negative_indices[:, 1]])),

    })
    return train_df, test_df

'''Read Dataset1'''


class KNN_Graph:
    def __init__(self, dis, k):
        self.dis = dis
        self.k = k
        self.graph = {i: [] for i in range(len(dis))}
        self.adjacency_matrix = np.zeros((len(dis), len(dis)))

    def calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def construct_graph(self):
        m,n=self.dis.shape
        for i in range(m):
            distances = self.dis[i]
            sort_dis=np.sort(distances)[::-1]
            nearest_neighbors = sort_dis[:self.k]

            for dist in nearest_neighbors:
                j=np.where(distances==dist)[0][0]
                self.graph[i].append(j)
                self.adjacency_matrix[i, j] = 1
                self.adjacency_matrix[j, i] = 1
        return self.graph

    def get_adjacency_matrix(self):
        return self.adjacency_matrix
def read_file1():
    '''one fold is used here as an example'''
    k=30
    lnc_mi = np.loadtxt("../data/dataset1/yuguoxian_lnc_mi.txt")
    mi_lnc = lnc_mi.T
    lnc_dis = np.loadtxt("../data/dataset1/lnc_dis_association.txt")
    mi_dis = np.loadtxt("../data/dataset1/mi_dis.txt")
    dis_sim = np.loadtxt("../data/dataset1/dis_fusion_sim.txt")
    lnc_sim = np.loadtxt("../data/dataset1/lnc_fusion_sim.txt")
    mi_sim = np.loadtxt("../data/dataset1/mi_fusion_sim.txt")

    '''dis_knn=KNN_Graph(dis_sim,k)
    lnc_knn=KNN_Graph(lnc_sim,k)
    mi_knn=KNN_Graph(mi_sim,k)
    dis_knn.construct_graph()
    lnc_knn.construct_graph()
    mi_knn.construct_graph()
    dis_sim=dis_knn.get_adjacency_matrix()
    lnc_sim=lnc_knn.get_adjacency_matrix()
    mi_sim=mi_knn.get_adjacency_matrix()'''

    lnc_dis_test_id = np.loadtxt("../data/dataset1/lnc_dis_test_id1.txt")
    mi_dis_test_id = np.loadtxt("../data/dataset1/mi_dis_test_id1.txt")
    mi_lnc_test_id = np.loadtxt("../data/dataset1/mi_lnc_test_id1.txt")
    return mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, mi_lnc_test_id

'''Read dataset2'''
def read_file2():

    '''one fold is used here as an example'''
    k=30
    di_lnc = pd.read_csv('../data/dataset2/di_lnc_intersection.csv', index_col='Unnamed: 0')
    di_mi = pd.read_csv('../data/dataset2/di_mi_intersection.csv', index_col='Unnamed: 0')
    mi_lnc = pd.read_csv('../data/dataset2/mi_lnc_intersection.csv', index_col='Unnamed: 0')

    lnc_dis = di_lnc.values.T
    mi_dis = di_mi.values.T
    mi_lnc = mi_lnc.values

    dis_sim = np.loadtxt("../data/dataset2/dis_fusion_sim.txt")
    lnc_sim = np.loadtxt("../data/dataset2/lnc_fusion_sim.txt")
    mi_sim = np.loadtxt("../data/dataset2/mi_fusion_sim.txt")
    dis_knn = KNN_Graph(dis_sim, k)
    lnc_knn = KNN_Graph(lnc_sim, k)
    mi_knn = KNN_Graph(mi_sim, k)
    dis_knn.construct_graph()
    lnc_knn.construct_graph()
    mi_knn.construct_graph()

    dis_sim = dis_knn.get_adjacency_matrix()
    lnc_sim = lnc_knn.get_adjacency_matrix()
    mi_sim = mi_knn.get_adjacency_matrix()

    lnc_dis_test_id = np.loadtxt("../data/dataset2/lnc_dis_test_id1.txt")
    mi_dis_test_id = np.loadtxt("../data/dataset2/mi_dis_test_id1.txt")
    mi_lnc_test_id = np.loadtxt("../data/dataset2/mi_lnc_test_id1.txt")        #
    return mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, mi_lnc_test_id






'''Zeroing of positive samples in the test set'''
def Preproces_Data (A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

'''Constructing adjacency matrix'''
def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim ):
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))

    dis_lnc_sim = np.hstack((lncRNA_disease.T,disease_sim, miRNA_disease.T))

    mi_lnc_dis = np.hstack((miRNA_lncRNA,miRNA_disease,miRNA_sim))

    matrix_A = np.vstack((lnc_dis_sim,dis_lnc_sim,mi_lnc_dis))          #
    return matrix_A

'''Normalisation'''
def lalacians_norm(adj):
    # adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)
    return norm_A


def get_adj(lnc_dis, mi_lnc, mi_dis):
    lnc_num = lnc_dis.shape[0]
    dis_num = lnc_dis.shape[1]
    mi_num = mi_lnc.shape[0]

    adj_matrix = np.zeros((lnc_num + dis_num + mi_num, lnc_num + dis_num + mi_num))


    adj_matrix[:lnc_num, lnc_num:lnc_num+dis_num] = lnc_dis

    adj_matrix[lnc_num:lnc_num+dis_num, :lnc_num] = lnc_dis.T


    adj_matrix[-mi_num:,:lnc_num] = mi_lnc


    adj_matrix[:lnc_num,-mi_num:] = mi_lnc.T


    adj_matrix[lnc_num:lnc_num+dis_num, -mi_num:] = mi_dis.T

    adj_matrix[-mi_num:, lnc_num:lnc_num+dis_num] = mi_dis

    return adj_matrix



'''gcn'''
class GCNConv(nn.Module):
    def __init__(self, in_size, out_size,):
        super(GCNConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, adj, features):
       out = torch.mm(adj, features)      # A*X，
       out = torch.mm(out,self.weight)    # A*X*W
       return out



def format_row(row_number, features, label):
    features_str = ','.join(str(feature) for feature in features)  # 将特征数组转换为逗号分隔的字符串
    return f"{row_number} {features_str} {label}"

def generate_case_id(rna_num,diease_id):
    # 第一列是0-230
    first_column = np.arange(rna_num)
    # 第二列是全部为5的元素
    second_column = np.full(rna_num, diease_id)
    # 合并两列以形成最终的数组
    case_id = np.column_stack((first_column, second_column))
    return case_id

if __name__ == '__main__':
    #assert 1==0
    #Hyperparameters
    Epoch = 500
    in_features = 1140
    # in_features = 1276  # num_lnc + num_dis + num_mi
    N_HID = 512
    out_features = 256
    LR = 0.0001

    mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_file1()
    #mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_file2()
    #mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_ndadata1()
    #mi_lnc, lnc_dis, mi_dis, dis_sim, lnc_sim, mi_sim, lnc_dis_test_id, mi_dis_test_id, lnc_mi_test_id = read_ndadata2()
    adj=get_adj(lnc_dis, mi_lnc, mi_dis)
    #np.save('../data/adj.npy', adj)
    lnc_dis_test_id=generate_case_id(240,62)
    mi_dis_test_id=generate_case_id(495,140)
    lnc_dis = Preproces_Data(lnc_dis,lnc_dis_test_id)
    mi_dis = Preproces_Data(mi_dis,mi_dis_test_id)
    mi_lnc = Preproces_Data(mi_lnc,lnc_mi_test_id)

    lnc_dis_sim = np.hstack((lnc_sim, lnc_dis))
    dis_lnc_sim = np.hstack((lnc_dis.T, dis_sim))
    samplefea=np.vstack((lnc_dis_sim,dis_lnc_sim))
    samplefea=lalacians_norm(samplefea)
    np.save('../data/samplefea.npy', samplefea)

    '''mi_dis_sim=np.hstack((mi_sim, mi_dis))
    dis_mi_sim = np.hstack((mi_dis.T, dis_sim))
    samplefea = np.vstack((mi_dis_sim, dis_mi_sim))
    samplefea = lalacians_norm(samplefea)
    np.save('../data/samplefea_MDA.npy', samplefea)
    '''
    matrix_A = construct_graph(lnc_dis,mi_dis,mi_lnc,lnc_sim,mi_sim,dis_sim)
    np.save('../data/matrix_A.npy', matrix_A)
    #edges = np.argwhere(matrix_A > 0)
    edges = np.argwhere(matrix_A > 0.5)

    # 由于矩阵是对称的，只取上三角部分，避免重复边
    edges = edges[edges[:, 0] < edges[:, 1]]
    edges_tensor = torch.tensor(edges, dtype=torch.int64)
    edges_tensor = edges_tensor.t()
    torch.save(edges_tensor, '../data/edges.pt')





    la_A = lalacians_norm(matrix_A)
    la_tensor=torch.tensor(la_A,dtype=torch.float32)
    torch.save(la_tensor, '../data/feature.pt')
    m,n=la_A.shape
    #dataset1 label
    label=np.array([0]*240+[1]*405+[2]*495)
    #dataset2 label
    #label = np.array([0] * 665+ [1] * 316 + [2] * 295)
    #ndadata1 label
    #label=np.array([0] * 56+ [1] * 121 + [2] * 464)
    #ndadata2 label
    #label = np.array([0] * 2189 + [1] * 1296 + [2] * 1596)
    label=torch.tensor(label,dtype=torch.int64)
    torch.save(label,'../data/label.pt')
    #train(la_A, Epoch, in_features, N_HID, out_features, LR)



