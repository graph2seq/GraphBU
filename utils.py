import os
import time
import re
import ast
import torch
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
MAX_TGT_LEN = 20

OPERATORS = {
    'Invert': '~',
    'Add': '+',
    'Sub': '-',
    'BitAnd': '&',
    'USub': '-',
    'BitOr': '|',
    'Mult': '*',
    'BitXor': '^',
    'UAdd': '+',
    'Pow': '**'
}


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

class ExprVisit(ast.NodeTransformer):
    '''
    Parsing an expression and generating the AST by the post-order traversal.
    For each expression, its variables must be a single character.
    '''
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.subtree_memo = []

    def merge_node(self, node):
        '''Determine whether the current node can be merged'''
        same_node = None
        if self.subtree_memo is None:
            return same_node
        cur_node_type = 'c' if hasattr(node, 'left') else 'unaryop'
        for p_node in self.subtree_memo:
            cur_p_node_type = 'binop' if hasattr(p_node, 'left') else 'unaryop'
            if cur_p_node_type != cur_node_type:
                continue
            elif cur_p_node_type == cur_node_type == 'binop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.left) == ast.dump(node.left) \
                    and ast.dump(p_node.right) == ast.dump(node.right):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)
            elif cur_p_node_type == cur_node_type == 'unaryop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.operand) == ast.dump(node.operand):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)

        return same_node

    def visit_BinOp(self, node):
        '''
        Scaning binary operators, such as +, -, *, &, |, ^
        '''
        self.generic_visit(node)
        # node.col_offset is unique identifer
        node_str = ast.dump(node.op) + str(node.col_offset)

        # Merge same subtrees or leaves
        same_node = self.merge_node(node)
        if same_node == None:
            self.subtree_memo.append(node)
            self.node_list.append(node_str)
        else:
            for idx in range(len(self.edge_list) - 1, -1, -1):
                if node_str == self.edge_list[idx][0] or \
                    node_str == self.edge_list[idx][1]:
                    del self.edge_list[idx]
            node_str = same_node

        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_UnaryOp(self, node):
        '''
        Scaning unary operators, such as - and ~
        '''
        self.generic_visit(node)
        node_str = ast.dump(node.op) + str(node.col_offset)
        
        # Merge same subtrees or leaves
        same_node = self.merge_node(node)
        if same_node == None:
            self.subtree_memo.append(node)
            self.node_list.append(node_str)
        else:
            for idx in range(len(self.edge_list) - 1, -1, -1):
                if node_str == self.edge_list[idx][0] \
                    or node_str == self.edge_list[idx][1]:
                    del self.edge_list[idx]
            node_str = same_node

        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_Name(self, node):
        '''
        Scaning variables
        '''
        self.generic_visit(node)
        # node.col_offset will allocate a unique ID to each node
        node_str = node.id

        if node_str not in self.node_list:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    # Use visit_Constant when python version >= 3.8
    # def visit_Constant(self, node):
    def visit_Num(self, node):
        '''
        Scaning numbers
        '''
        self.generic_visit(node)
        node_str = str(node.n)

        if node_str not in self.node_list:
            self.node_list.append(node_str)
        node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
        self.edge_list.append([node_str, node_parent_str])

        return node

    def get_result(self):
        return self.node_list, self.edge_list


def expr2graph(expr):
    '''
    Convert a expression to a MSAT graph.

    Parameters:
        expr: A string-type expression.

    Return:
        node_list:  List for nodes in graph.
        edge_list:  List for edges in graph.
    '''
    ast_obj = ast.parse(expr)

    for node in ast.walk(ast_obj):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    vistor = ExprVisit()
    vistor.visit(ast_obj)
    node_list, edge_list = vistor.get_result()
    return node_list, edge_list


def repl(integer):    #例如将整数 123 转换成 (1*10**2+2*10**1+3)。
    integer = integer.group(0)
    if len(integer) == 1: 
        return integer

    expr, length = '', len(integer)

    for i in range(length):
        if integer[i] == '0': continue
        if i == length - 1: expr += '+' + integer[i]
        elif i == length - 2: expr += ('' if length == 2 else '+') + integer[i] + '*10'
        else: expr += ('+' if i else '') + integer[i] + f'*10**{length-1-i}'

    return  '(' + expr + ')'
def num_decompose(expr):
    return re.sub(r'\d+\.?\d*', repl, expr)      #对数字进行处理，当匹配到连续两位数字时 即多为整数或小数，将其转化 用上面的repl处理


class GraphExprDataset(InMemoryDataset):
    '''
    Base class of our dataset.
    '''
    def __init__(self, root, dataset):
        # self.dataset = dataset
        # self.qst_vocab = {}
        # self.ans_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        # self.qst_vocab['10'] = len(self.qst_vocab)
        # self.qst_vocab['**'] = len(self.qst_vocab)
        # super().__init__(root)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.dataset = dataset

        super().__init__(root)
        self.data, self.slices,self.qst_vocab,self.ans_vocab= torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'dataset/raw/{self.dataset}.csv'
    
    @property
    def processed_file_names(self):
        return f'{self.dataset}.pt'
    
    def download(self):
        pass

    def process(self):
        data_list = []
        self.qst_vocab = {}
        self.ans_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        #pad表示填充，eos表示表达式结束
        self.qst_vocab['10'] = len(self.qst_vocab)
        self.qst_vocab['**'] = len(self.qst_vocab)
        df = pd.read_csv(self.raw_file_names, header=None, nrows=None)  #读取原始数据集
        #数据集样式
        # 0      9 * ((-5 * (x & y)) & (-3 * (x & ~y))) + 3 * ((-5 * (x & y)) & ~(-3 * ...            7 * ~x-5 * x
        # 1       -5 * ((-8 * (x & y)) & ((x & ~y)))-10 * ((-8 * (x & y)) & ~((x & ~...               - 4 * x - 6 * ~y
        # 2       -2 * ((x & y) & (-6 * (x & ~y))) + 2 * ((x & y) & ~(-6 * (x & ~y))) - ...           - 7 * ~y + ~x
        max_tgt_len = float('-inf')
        
        # 每一行就是一个样本
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # row[0] is source expression, which will be transformed into a graph
            raw_qst, raw_ans = str(row[0]), str(row[1])          #qst是需要化简的表达式（特征）   ans是化简后的表达式（标签）
            expr = num_decompose(raw_qst).replace('+-', '-').replace('--', '+')
            
            for c in expr:    #对qst和ans编码
                # qst_vocab因为会转换成树，不需要括号
                if c != '(' and c != ')' and c not in self.qst_vocab:
                    self.qst_vocab[c] = len(self.qst_vocab)
            for c in raw_ans:
                if c not in self.ans_vocab:
                    self.ans_vocab[c] = len(self.ans_vocab)

            x, edge_index = self._generate_graph(expr)        #特征矩阵和邻接矩阵
            #print(expr)
            # raw_ans is target, which will be represented as a one-hot vector
            # if len(raw_ans) <= MAX_TGT_LEN:
            y = [self.ans_vocab[c] for c in raw_ans]
            y.insert(0, self.ans_vocab['<sos>'])
            y.append(self.ans_vocab['<eos>'])

            max_tgt_len = max(max_tgt_len, len(y))

            y = torch.tensor(y, dtype=torch.long)
            #y: tensor([1, 7, 10, 4, 5, 6, 12, 3, 4, 5, 11, 2])
            # Fed the graph and the label into Data
            data = Data(x=x, edge_index=edge_index, y=y, src=raw_qst)
            #print("data",data)
            # data = Data(x=x_s, edge_index=edge_index_s)
            data_list.append(data)

        # Pad the target
        for data in data_list:
            padding = torch.zeros(max_tgt_len - data.y.shape[0], dtype=torch.long)
            # padding = torch.tensor([self.ans_vocab['<pad>']] * (max_tgt_len - data.y.shape[0]), dtype=torch.long)
            data.y = torch.cat((data.y, padding), dim=0)
            padding = torch.zeros((data.x.shape[0], len(self.qst_vocab)-data.x.shape[1]), dtype=torch.float)
            data.x = torch.cat((data.x, padding), dim=1)

        self.max_tgt_len = max_tgt_len
        # exit(0)
        data, slices = self.collate(data_list)
        # G = to_networkx(data, to_undirected=True)
        # visualize_graph(G, color=data.y)
        torch.save((data, slices,self.qst_vocab,self.ans_vocab), self.processed_paths[0])
        #torch.save((data, slices), self.processed_paths[0])

    def _generate_graph(self, expr):
        try:
            #print(self.qst_vocab)
            node_list, edge_list = expr2graph(expr)
            # node_list['5', 'USub()0', '8', 'USub()5', 'x', 'y', 'BitAnd()9', 'Mult()5', 'Invert()19', 'BitAnd()17', 'BitAnd()4', 'Mult()0', '1', '10', 'Mult()26', 'Invert()44', 'BitAnd()33', 'Mult()25', 'Sub()0', '4', 'BitXor()58', 'Mult()55', 'Add()54', '7', 'Mult()79', 'Add()78', '3', 'BitOr()96', 'Invert()94', 'Mult()92', 'Sub()91', 'Mult()118', 'Add()117', 'Add()123', 'Invert()131', 'Mult()122', 'Sub()121', 'BitOr()142', 'Invert()140', 'Mult()138', 'Add()137', 'BitOr()151', 'Invert()149', 'Mult()147', 'Add()146']
            # edge_list[['5', 'USub()0'], ['USub()0', 'Mult()0'], ['8', 'USub()5'], ['USub()5', 'Mult()5'], ['x','BitAnd()9'], ['y', 'BitAnd()9'], ['BitAnd()9', 'Mult()5'], ['Mult()5', 'BitAnd()4'], ['x', 'BitAnd()17'], ['y','Invert()19'], ['Invert()19', 'BitAnd()17'], ['BitAnd()17', 'BitAnd()4'], ['BitAnd()4', 'Mult()0'], ['Mult()0','Sub()0']]
        except:
            raise ValueError(expr)

        node_feature = []

        for node in node_list:
            tag = node.split('()')[0]
            if tag in OPERATORS: 
                tag = OPERATORS[tag]

            feature = [0] * len(self.qst_vocab)
            feature[self.qst_vocab[tag]] = 1
            
            node_feature.append(feature)

        COO_edge_idx = [[], []]
        for edge in edge_list:
            s_node, e_node = node_list.index(edge[0]), node_list.index(edge[1])
            COO_edge_idx[0].append(s_node), COO_edge_idx[1].append(e_node)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(COO_edge_idx, dtype=torch.long)
        #print(x)
        #print(edge_index)
        # ((((y + 1) + ~((t & ~z) + z)) + 1) | (((~z | y) + z) + 1)) + (
        #             (((y + 1) + ~((t & ~z) + z)) + 1) & (((~z | y) + z) + 1))
        # x    one-hot向量，对应了前面的resource vocab
        # {'10': 0, '**': 1, '-': 2, '5': 3, '*': 4, '8': 5, 'x': 6, '&': 7, 'y': 8, '~': 9, '1': 10, '0': 11, '+': 12, '4': 13, '^': 14, '7': 15, '3': 16, '|': 17, '2': 18, '6': 19, 't': 20, 'z': 21}
        # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 1., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 1.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.],
        #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        #          0., 0., 0., 0.]])
        # tensor([[0, 1, 2, 3, 4, 5, 6, 4, 7, 8, 9, 1, 10, 5, 0, 11, 4, 12,
        #          1, 13, 14, 10, 13, 15],
        #         [2, 2, 9, 6, 5, 6, 7, 7, 8, 9, 10, 10, 14, 11, 11, 12, 12, 13,
        #          13, 14, 16, 15, 15, 16]])
        return x, edge_index


