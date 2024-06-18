import torch
import torch.nn as nn
import numpy as np
#定义编码器，字典的大小为10，要把token编码成128维的向量
embedding=nn.Embedding(10,128)
#定义 transformer，模型维度为128（也就是词向量的维度）
transformer=nn.Transformer(d_model=128)
#定义一个源句子，可以想象成是（<bos>,我，爱，吃，肉，和，菜，<eos>,<pad>,<pad>）
src=torch.LongTensor([[0,3,4,5,6,7,8,1,2,2]])
print(src.shape)
#   定义目标句子，可以想象成是（<bos>,I like eat meat and vegetable <eos> <pad>）
tgt=torch.LongTensor([[0,3,4,5,6,7,8,1,2]])
#将token编码后送给transformer，暂时不加位置编码
outputs=transformer(embedding(src).reshape(-1,1,128),embedding(tgt).reshape(-1,1,128))

print(outputs.shape)
