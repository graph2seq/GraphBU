import  torch
import torch_geometric
import torch.nn as nn
import numpy as np
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from scipy.sparse import *
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from torch_sparse import SparseTensor
#torch.set_printoptions(threshold=np.sys.maxsize)
#input_dim:26  output_dim:28
class GNN(MessagePassing):
    def __init__(self,in_channels,out_channels):
        super(GNN, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.lin=Linear(in_channels,out_channels)#weight_initializer权重初始化

    def forward(self,x,edge_index):
        x=self.lin(x)
        #print(edge_index.T.dtype==torch.long)
        # print(edge_index)
        # print(x.shape)
        # print(edge_index.shape)
        # print(type(edge_index))
        #edge_index,_=add_self_loops(edge_index,num_nodes=x.shape[0])      #MARK:暂时先不加自环
        edge_index.requires_grad=False
        son=edge_index[0]              #子节点
        father=edge_index[1]           #父节点
        leaf=[]                        #叶子节点
        num_vertex=x.shape[0]          #总节点数
        num_edge=len(father)           #总边数
        vertex = [0 for i in range(0, num_vertex)]        #用于记录当前状态下各个顶点的子节点个数
        deg = [0 for i in range(0, num_vertex)]           #用于记录各个节点的度数
        person = [[] for i in range(0, num_vertex)]       #记录每个顶点有哪些父亲节点
        for i in range(0,num_edge):
            vertex[father[i]]+=1        #
            deg[father[i]]+=1
            person[son[i]].append(father[i])
        for i in range(num_vertex):
            if vertex[i]==0:
                leaf.append(i)
        head=0
        while(head<num_vertex):
            temp=leaf[head]           #当前操作的子节点

            for i in person[temp]:    #i是该叶子节点的所有父亲节点，这一步就是将该子节点的信息，传递给它的所有父亲节点

                x[i]+=x[temp]/deg[i]        #除以度数
                vertex[i]-=1             #同时将该叶子节点剪除
                if vertex[i]==0:
                    leaf.append(i)
            head+=1
        return x








    # def forward(self,x,edge_index):
    #     print(edge_index.T.)
    #     h_v = self.MLP_v(x)  # shape:2630,128
    #     flag=0
    #     #先找父节点
    #     for source,target in edge_index:
    #         if source==flag:
    #             feature_com=torch.cat((h_v[target],h_v[source]),dim=1)
    #             feature_mlp=self.MLP_edge()
    #             feature
    #
    #     print(x)

    #方法二：直接用普通的方法实现——复杂度很高
    # def forward(self,x,edge_index):
    #     #首先对邻接矩阵添加自环
    #     #shape x(2630,26)
    #     edge_index,_=add_remaining_self_loops(edge_index)
    #     team=torch.ones(edge_index.shape[1])
    #     edge_index = csr_matrix((team, (edge_index[0].cpu(),edge_index[1].cpu())), shape=(x.shape[0], x.shape[0]))
    #     #print(edge_index.dtype==torch.long)   #true
    #     #print(edge_index.todense().shape)    #2630,2630
    #     edge_index=edge_index.todense()
    #     # [[1. 1. 0.... 0. 0. 0.]
    #     #  [0. 1. 0.... 0. 0. 0.]
    #     # [0.0.1....0.0.0.]                  #2630,2630
    #
    #     h_v=self.MLP_v(x)                    #shape:2630,128
    #
    #     for vertex in range(h_v.shape[0]):
    #         #首先选中父亲节点和子节点：
    #         node_fathers=edge_index[vertex].A#shape:list:2630
    #         node_fathers = np.array(node_fathers, dtype=bool)  #[[ True  True False ... False False False]] 作为MASK索引矩阵行
    #         node_sons=edge_index[:,vertex].reshape(1,-1).A.tolist()#shape:list:2630
    #         node_sons = np.array(node_sons, dtype=bool)  #[[ True False False ... False False False]] 作为MASK索引矩阵行
    #         feature_fathers=h_v[node_fathers]             #shape:*，128 #来自父亲节点的特征
    #         feature_sons=h_v[node_sons]                   #shape:*，128 #来自子节点的特征
    #         num_fathers=feature_fathers.shape[0]          #MARK：父节点的数量
    #         num_sons=feature_sons.shape[0]                #MARK：子节点的数量
    #         #聚集信息
    #         feature_self_father=torch.tensor((h_v[vertex].unsqueeze(0).tile(num_fathers,1))).clone().detach().requires_grad_(True)
    #         feature_self_son=torch.tensor((h_v[vertex].unsqueeze(0).tile(num_sons,1))).clone().detach().requires_grad_(True)
    #         feature_fathers_com=torch.cat((feature_fathers,feature_self_father),dim=1)  #shape:*,256
    #         feature_sons_com=torch.cat((feature_sons,feature_self_son),dim=1)         #shape:*,256
    #         feature_fathers_mlp=self.MLP_edge(feature_fathers_com)              #MARK：从父亲节点收集的信息
    #         feature_sons_mlp=self.MLP_edge(feature_sons_com)                     #MARK：从子节点收集的信息
    #
    #         #传递
    #         feature_fathers_mlp=feature_fathers_mlp.mean(axis=0)/num_fathers
    #         feature_sons_mlp=feature_sons_mlp.mean(axis=0)/num_sons
    #         feature_all_com=torch.cat((h_v[vertex].unsqueeze(0),feature_fathers_mlp.unsqueeze(0)),dim=1)
    #         feature_all_com=torch.cat((feature_all_com,feature_sons_mlp.unsqueeze(0)),dim=1)  #shape:*,384
    #         feature_all_mlp=self.MLP_aggr(feature_all_com)                       #shape:*,128
    #         #更新
    #         h_v[vertex]=h_v[vertex]+feature_all_mlp
    #     #print(h_v.shape)    #torch.Size([2630, 128])
    #
    #     return h_v

    # def forward(self,x,edge_index):
    #     #首先对邻接矩阵添加自环
    #     #shape x(2678,26)
    #     edge_index=edge_index.t()
    #     print(edge_index)
    #     print(edge_index.dtype==torch.long)   #shape(2,6567)
    #     # tensor([[   0,    1,    2,  ..., 2675, 2676, 2677],
    #     # [   3,    2,    3,  ..., 2675, 2676, 2677]], device='cuda:0')
    #     h_temp=self.linear1(x)
    #     #print(h_temp) shape(2678,256)
    #
    #     h_v=self.linear2(h_temp)
    #     #shape(2678,128)
    #     num_parents=0  #每个节点的双亲数
    #     num_sons=0     #每个节点的儿子数
    #     com_son_sum=torch.zeros(1,128)
    #     com_parents_sum=torch.zeros(1,128)
    #     com_parents_sum=com_parents_sum.to('cuda')
    #     com_son_sum=com_son_sum.to('cuda')
    #     for vertex in range(h_v.shape[0]):        #对于每一个节点
    #         #print("x[vertex]: ",x[vertex])
    #         for source,target in edge_index:   #对于每一条边
    #             if source==vertex:
    #                 com=torch.cat((h_v[target].unsqueeze(0),h_v[vertex].unsqueeze(0)),dim=1)    #将两个节点的特征组合在一起
    #                 com_out=self.MLP_edge(com)
    #                 com_parents_sum+=com_out
    #                 num_parents+=1
    #             if target==vertex:
    #                 com=torch.cat((h_v[source].unsqueeze(0),h_v[vertex].unsqueeze(0)),dim=1)
    #                 com_out = self.MLP_edge(com)
    #                 com_son_sum+=com_out
    #                 num_sons+=1
    #
    #         com_parent_aggr=com_parents_sum/num_parents
    #         # print(com_parent_aggr.shape)
    #         # print(h_v[vertex].unsqueeze(0).shape)
    #         com_son_aggr=com_son_sum/num_sons
    #         aggr_son_parents_0=torch.cat((h_v[vertex].unsqueeze(0),com_parent_aggr),dim=1)
    #         aggr_son_parents=torch.cat((aggr_son_parents_0,com_son_aggr),dim=1)
    #         aggr_mlp=self.MLP_aggr(aggr_son_parents)
    #         h_v[vertex]=h_v[vertex]+aggr_mlp
    #     print(h_v.shape)
    #
    #     return h_v








