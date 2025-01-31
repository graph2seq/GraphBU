import os
import z3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import gradcheck
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import GraphExprDataset
from models.gnn import Encoder, Decoder, Graph2Seq

# from models.bert import Encoder,Decoder,Seq2Seq

SEED = 1234  # 固定随机种子，保证训练的结果一致。
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda')


class Model:  # 定义模型
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _count_paras(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算有多少参数需要梯度

    def build_model(self):
        hid_dim = 512
        # dec_emb_dim = 256       #解码器embedding的维度
        dec_emb_dim = 512  # 当使用transformer时需要和编码器的维度对应 所以是512
        dropout = 0.5
        device = DEVICE

        encoder = Encoder(input_dim=self.input_dim, hid_dim=hid_dim)  # 输出维度512
        # decoder = Decoder(output_dim=self.output_dim, emb_dim=dec_emb_dim, hid_dim=hid_dim)  #输入input的维度256
        decoder = Decoder(output_dim=self.output_dim, emb_dim=dec_emb_dim)  # 输入input的维度256
        model = Graph2Seq(encoder, decoder, device).to(device)

        print()
        print('=======================================')
        print(f'Input dim:  {self.input_dim}')
        print(f'output dim: {self.output_dim}')
        print(f'Hidden dim: {hid_dim}')
        print(f'Count trainable paras: {self._count_paras(model):,}')

        return model


class Trainer:
    def __init__(self, model, epochs, ignore_index,vocab):
        self.model = model
        self.epochs = epochs
        self.device = DEVICE
        self.ver_vocab=list(vocab)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)                                    #MARK:改了
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.02, patience=6, verbose=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)  # 交叉熵损失函数

    def _train(self, data_loader):
        self.model.train()
        epoch_loss = 0

        for i, data in enumerate(data_loader):
            # out = [tgt_len, batch_size, vocab_size]

            data.to(self.device)
            data.requires_grad=True

            # print("data:",data.y)
            out = self.model(data,mode="train")  # Perform a single forward pass.
            out_dim = out.shape[-1]
            out = out[:-1].reshape(-1, out_dim)
            tgt = data.y.reshape(data.num_graphs, -1).t().to(self.device)
            tgt = tgt[1:].reshape(-1)       #
            loss = self.criterion(out, tgt)  # Compute the loss.
            loss.backward()  # Derive gradients.
            #print([x.grad for x in self.optimizer.param_groups[0]['params']])
            #print([x.grad for x in self.model.decoder.parameters()])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()  # Update parameters based on gradients.

            self.optimizer.zero_grad()  # Clear gradients.

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def _evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data.to(self.device)
                out = self.model(data,"vaild")  # trun off teacher forcing
                # tgt = [tgt_len, batch size]
                # output = [tgt_len, batch size, output dim]
                out_dim = out.shape[-1]
                out = out[:-1].reshape(-1, out_dim)
                tgt = data.y.reshape(data.num_graphs, -1).t().to(self.device)
                tgt = tgt[1:].reshape(-1)
                # tgt = [(tgt len - 1) * batch size]
                # output = [(tgt len - 1) * batch size, output dim]
                #print("out",out.argmax(1).shape)
                #print("tgt",tgt.shape)
                loss = self.criterion(out, tgt)
                epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def run(self, train_iter, valid_iter, model_name):
        print()
        print('Training...')
        print('=======================================')
        best_valid_loss = float('inf')
        best_train_loss = float('inf')

        # param used in early stoping
        es_param = 'train_loss'

        # patience for early stopping
        patience = 20
        trigger = 0
        total_train_time = 0

        for epoch in range(self.epochs):
            start_time = time()
            print("开始训练")
            #train_loss=0
            train_loss = self._train(train_iter)
            total_train_time += time() - start_time
            print("评价")
            valid_loss = self._evaluate(valid_iter)
            print(
                f'Epoch {epoch + 1:>3d}: train loss: {train_loss:.4f}, val loss: {valid_loss:.4f}, time: {time() - start_time:.4f}')
            writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trigger = 0
                torch.save(model.state_dict(), f'saved_models/{model_name}.pt')
            else:
                trigger += 1
                if trigger >= patience:
                    break
            self.scheduler.step(valid_loss)

        print(f'Best valid loss: {best_valid_loss:.4f}')
        print(f'Training time per epoch: {total_train_time / self.epochs:.4f}')


class Verifer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ver_vocab = list(vocab)
        self.device = DEVICE

    def predict(self, data, model, max_len=120):
        model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(self.device)
            context = model.encoder(x, edge_index, batch)
            context = context.unsqueeze(0)
            softmax=nn.Softmax(dim=2)
        tgt_idx = [self.vocab['<sos>']]

        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_idx).unsqueeze(1).to(self.device)
            #print(tgt_idx)
            with torch.no_grad():
                #context=torch.cat((context,context[-1].unsqueeze(0)),dim=0)
                output = model.decoder(tgt_tensor, context)
                output=softmax(output)
            top1=output[-1].argmax(dim=1)
            tgt_idx.append(top1)

            if top1 == self.vocab['<eos>']:
                break

        tgt_tokens = [self.ver_vocab[i] for i in tgt_idx]
        return ''.join(tgt_tokens[1:-1])

    def z3_verify(self, src, tgt):
        x, y, z, t, a, b, c, d, e, f = z3.BitVecs('x y z t a b c d e f', 4)
        s = z3.Solver()
        try:
            s.add(eval(src) != eval(tgt))
        except:
            return False
        if s.check() == z3.unsat:
            return True
        return False

    def count_acc(self, dataset, model):
        print()
        print('Verification...')
        print('=======================================')

        num_total_equal, num_seman_equal = 0, 0
        total_infer_time = 0

        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            tgt = ''.join([self.ver_vocab[i] for i in data.y]).split('<eos>')[0][5:]

            start_time = time()
            pred = self.predict(data, model)

            pred_time = time() - start_time
            total_infer_time += pred_time
            print(f'{data.src},{tgt},{pred},{pred_time:.4f},{pred == tgt}')
            print("====success!!!!===")
            if pred == tgt:
                num_total_equal += 1
            elif self.z3_verify(pred, tgt):
                num_seman_equal += 1

        size = len(dataset)
        print(f'Validation set size: {size}')
        print(f'Inference time per sample: {total_infer_time / size:.4f}')
        print(f'Formal equal count:\t{num_total_equal}/{size}')
        print(f'Semantic equal count:\t{num_seman_equal}/{size}')
        print(f'Accuracy:\nWithout semantic equal\t:\t{num_total_equal / size:.4f}')
        print(f'\tWith semantic equal:\t{(num_total_equal + num_seman_equal) / size:.4f}')


if __name__ == '__main__':
    """
    Accept 5 parameters:
        type: 'ast' or 'dag'.
        dataset: dataset name which includes 'mba', poly1', and 'poly6'.
        train: whether to train the model.
        batch_size: batch size.
        epochs: epochs
    """
    parser = ArgumentParser(description='GraphMR')  # 首先设置用户参数
    # parser.add_argument('--dataset', type=str, default='np_mix', help='dataset, MBA, POLY1 or POLY6')
    parser.add_argument('--dataset', type=str, default='poly6', help='dataset, MBA, POLY1 or POLY6')
    parser.add_argument('--train', type=int, default=1, help='whether to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='size of each mini batch')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')

    args = parser.parse_args()

    # === 1. Get dataset ==============================
    # Before creating dataset, destination folder should be empty
    #os.system(f'rm -rf dataset/processed/*')
    writer = SummaryWriter()
    dataset = GraphExprDataset(root='dataset', dataset=args.dataset)
    train_valid_split = int(0.9 * len(dataset)) # 划分数据集
    train_set = dataset[:train_valid_split]
    valid_set = dataset[train_valid_split:]

    print()
    print(f'Dataset: {args.dataset}')
    print(f'===================================')
    print(f'Number of graphs: {len(dataset)}')  # 包含了多少公式
    print(f'Number of features: {dataset.num_features}')  # 每一个公式为23特征
    print(f'Number of training graphs: {len(train_set)}')  # 训练集大小
    print(f'Number of valid graphs: {len(valid_set)}')  # 验证集大小
    print(f'Source vocab: {dataset.qst_vocab}')  # 资源字典
    print(f'Target vocab: {dataset.ans_vocab}')  # 目标字典
    print(f'Batch size: {args.batch_size}')  # batch
    print(f'Epochs:{args.epochs}')  # epoch
    # Source
    # vocab: {'10': 0, '**': 1, '-': 2, '5': 3, '*': 4, '8': 5, 'x': 6, '&': 7, 'y': 8, '~': 9, '1': 10, '0': 11, '+': 12,
    #         '4': 13, '^': 14, '7': 15, '3': 16, '|': 17, '2': 18, '6': 19, 't': 20, 'z': 21, '9': 22}
    # Target
    # vocab: {'<pad>': 0, '<sos>': 1, '<eos>': 2, '-': 3, '4': 4, '*': 5, 'x': 6, '6': 7, '~': 8, 'y': 9, '7': 10,
    #         '+': 11, '(': 12, ')': 13, 't': 14, '|': 15, 'z': 16, '&': 17, '3': 18, '2': 19, '^': 20, '8': 21, '5': 22,
    #         '9': 23}

    # === 2. Get model ==============================
    input_dim = dataset.num_node_features  # input_dim：26
    # print("ans_vab!!",dataset.ans_vocab)
    # print("len ans vab",len(dataset.ans_vocab))
    output_dim = len(dataset.ans_vocab)  # output_dim：28
    model = Model(input_dim, output_dim).build_model()
    model_name = 'graphmr_' + args.dataset + "final"
    # === 3. Training ==============================
    if args.train:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

        trainer = Trainer(model, args.epochs, dataset.ans_vocab['<pad>'],dataset.ans_vocab)
        trainer.run(train_loader, valid_loader, model_name)

    # === 4. Verifing ==============================
    model.load_state_dict(torch.load(f'saved_models/{model_name}.pt'))
    verifer = Verifer(dataset.ans_vocab)
    verifer.count_acc(valid_set, model)

    writer.close()
