# 更短的session？
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import utils as nn_utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 先获得特征向量
global user
global user_rate
global rate_latest
global item
global class_dict
global gender_dict
global train_data, train_label, test_data, test_label, valid_data, valid_label

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action='store_false')
parser.add_argument("--model_path", default='D:\dataproject\\checkpoint')
parser.add_argument("--session_length", type=int, default=40)
parser.add_argument("--train_mode", choices=['test_next', 'test_last'], default='test_next')
parser.add_argument("--train_model", choices=['gru', 'rnn'], default='gru')
parser.add_argument("--learn_rate", type=int, default=0.005)
# parser.add_argument("--loss_type", choices=['CrossEntropy', 'SampledCrossEntropy', 'TOP1'], default='CrossEntropy')
parser.add_argument("--loss_type", choices=['CrossEntropy', 'SampledCrossEntropy', 'TOP1'], default='TOP1')
parser.add_argument("--layer_dim", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=5)
args = parser.parse_args()


# 使用user_rate 作为输入，rate_latest作为目标输出
user_rate = [[-1] for __ in range(6040)]  # 是用户按时间看的电影的向量，rnn使用
user_time = [[-1] for __ in range(6040)]  # 是用户看电影的时间，排序用
rate_latest = [0 for _ in range(6040)]  # rnn网络的目标是预测最后一部看的电影
rate_tuple = [[-1] for _ in range(6040)]  # 为了排序，tuple[i] = (time，movie) 代表time时刻用户i看了movie号电影
device = 'cuda'
mode = 'test_next'
session_mode = 'long'  # or cut
# 可能是因为把每个用户当作一个session，没有考虑到用户喜好的变化？
# 试试把一天当作一个session
# 还是太长，试试确定长度
if args.session_length != -1:
    session_size = args.session_length
    # session_mode = 'long'

loss_type = args.loss_type



def load_data():
    # global user
    global user_rate
    global rate_latest
    global user_time
    global maxlen
    print('heyyyyyy')
    path3 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path3, 'rb')
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        cu = int(rat_data[0]) - 1
        ci = int(rat_data[1]) - 1
        cr = int(rat_data[2])
        ct = int(rat_data[3])  # 时间
        if rate_tuple[cu][0] == -1:
            rate_tuple[cu][0] = (ct, ci)
        else:
            rate_tuple[cu].append((ct, ci))

    # print(rate_tuple[339])

    if session_mode == 'cut':
        user_rate = []
        rate_latest = []
        for i in range(6040):
            rate_tuple[i].sort()
            # 切分成更小的session
            rate_earliest = rate_tuple[i][0][0]
            cur_rating = 0
            user_earliest = rate_earliest
            while cur_rating < len(rate_tuple[i]):
                cur_session = [rate_tuple[i][item][1] for item in range(len(rate_tuple[i]))
                               if cur_rating <= rate_tuple[i][item][0] < cur_rating + session_size]
                user_earliest += session_size
                cur_rating += len(cur_session)
                if len(cur_session) > 1:
                    user_rate.append(cur_session)
                    rate_latest.append(cur_session[-1])
    else:
        for i in range(6040):
            rate_latest[i] = rate_tuple[i][-1][1]
            for movs in range(len(rate_tuple[i])):
                if user_rate[i][0] == -1:
                    user_rate[i][0] = rate_tuple[i][movs][1]
                else:
                    user_rate[i].append(rate_tuple[i][movs][1])
            # print(user_rate[i])

    print(len(user_rate))
    print('finished generating rate features')


# 构造数据集
def prepare_data():
    global user
    global user_rate
    global rate_latest
    global item
    global train_data
    global train_label
    global test_data
    global test_label
    global valid_data
    global valid_label

    total_set = list(zip(user_rate, rate_latest))
    random.shuffle(total_set)
    total_data, total_label = zip(*total_set)
    total_data = list(total_data)
    total_label = list(total_label)
    test_size = int(len(total_data) / 10)
    valid_size = int(len(total_data) / 10)
    test_data = total_data[:test_size]
    test_label = total_label[:test_size]
    valid_data = total_data[test_size:test_size + valid_size]
    valid_label = total_label[test_size:test_size + valid_size]
    train_data = total_data[test_size + valid_size:]
    train_label = total_label[test_size + valid_size:]
    # print(len(test_label))
    # print(len(valid_label))
    # print(len(train_data))

    # 取最后一个看的电影作为label，或者取每部电影的下一部
    if mode == 'test_last':
        for datas in range(test_size):
            test_data[datas] = test_data[datas][:-1]  # 将最后一个评分的项抹除
        for datas in range(valid_size):
            valid_data[datas] = valid_data[datas][:-1]
        for datas in range(len(train_data)):
            train_data[datas] = valid_data[datas][:-1]
    else:
        for datas in range(test_size):
            test_label[datas] = test_data[datas][1:]
            test_data[datas] = test_data[datas][:-1]  # 将最后一个评分的项抹除
        for datas in range(valid_size):
            valid_label[datas] = valid_data[datas][1:]
            valid_data[datas] = valid_data[datas][:-1]
        for datas in range(len(train_data)):
            train_label[datas] = train_data[datas][1:]
            train_data[datas] = train_data[datas][:-1]

    print('finished test and valid preparing, test_size=')
    return test_data, test_label, train_data, train_label, valid_data, valid_label


class Moviedata(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = []
        self.count_len(data)

    def count_len(self, data):
        for ratings in data:
            cur_len = len(ratings)
            self.length.append(cur_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.length[item]


def collate_fn(batch):
    lengths = [f[2] for f in batch]
    max_len = max(lengths)
    ranks = list(np.argsort(lengths))
    ranks.reverse()
    lengths.sort(reverse=True)
    input_ids = [batch[i][0] + [0] * (max_len - len(batch[i][0])) for i in ranks]
    if mode == 'test_last':
        input_mask = [[0.0] * len(batch[i][0]) + [0.0] * (max_len - len(batch[i][0])) for i in ranks]
        for i in ranks:
            input_mask[i][len(batch[i][0]) - 1] = 1.0
    else:
        input_mask = [[1.0] * len(batch[i][0]) + [0.0] * (max_len - len(batch[i][0])) for i in ranks]

    labels = [batch[i][1] for i in ranks]
    # print(labels)
    labels_flatten = []
    for i in ranks:
        labels_flatten.extend(batch[i][1])

    return input_ids, labels, labels_flatten, lengths, input_mask

class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda=False):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        # 不是很能理解SampledCrossEntropy
        if loss_type == 'CrossEntropy':
            self._loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'SampledCrossEntropy':
            self._loss_fn = SampledCrossEntropyLoss(True)
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)

class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()
    def forward(self, logit):
        """
        Args:
            logit (B x min_len): Variable that stores the logits for the items in the mini-batch
            The first dimension corresponds to the batches, and the second
            dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss

class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self, use_cuda):
        """
        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(1)
        target = Variable(torch.arange(batch_size).long())
        if self.use_cuda:
            target = target.cuda()

        return self.xe_loss(logit, target)


class RNN(nn.Module):

    def __init__(self, movie_num, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        :param movie_num: 输入数据的维度
        :param embedding_dim：词向量的维度
        :param hidden_dim: RNN神经元个数
        :param layer_dim: RNN的层数
        :param output_dim: 隐藏层输出的维度
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(movie_num, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, output_dim)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, seq_length, mask):
        """
        :param x: [batch, time_step, ]
             out: [batch, time_step, hidden_dim]
             h_n: [layer_dim, batch, hidden_dim]
          logits: [batch, time_step, movie_num]
        perd_tag: [batch, time_step]
        :return: 最后一个时间点的out输出
        """

        embeds = self.embedding(x)
        # print(embeds)
        pack = nn_utils.rnn.pack_padded_sequence(embeds, seq_length, enforce_sorted=False, batch_first=True)
        out, h_n = self.rnn(pack, None)
        unpacked, bz = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        # print(unpacked.size())
        logits = self.fc1(unpacked)
        ranks = logits
        # print(logits.size())
        pred_tag = torch.argmax(logits, dim=-1)
        # print(pred_tag.size())

        # Compute loss. Pad token must be masked before computing the loss.
        logits = logits.view(-1, movie_num)[mask.view(-1) == 1.0]
        # print(logits.size())
        # print(labels.size())
        loss = self.loss(logits, labels.view(-1))

        return loss, pred_tag, ranks


class GRU(nn.Module):
    def __init__(self, movie_num, embedding_dim, hidden_dim, layer_dim, output_dim):
        '''

        :param movie_num: 输入数据的维度
        :param embedding_dim: 词向量的维度
        :param hidden_dim: GRU神经元个数
        :param layer_dim: 层数
        :param output_dim: 输出的维度
        '''
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        print(layer_dim)
        self.embedding = nn.Embedding(movie_num, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.final_activation = nn.Tanh()
        if loss_type == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = LossFunction(loss_type=loss_type, use_cuda=True)
        # self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, seq_length, mask):
        '''

        :param x: 输入数据
        :param labels: 标签（最后一个时刻的输出）
        :param seq_length: batch中各向量的真实长度
        :param mask: 取输出，计算loss（next/last）
        :return: loss 交叉熵，pred_tag 预测的最后一部电影

        output: [batch, seq_len, hidden_size]
        h_n: [batch, num_s, hidden_size]
        unpacked: [batch, seq_len, hidden_size]
        logits: [batch, seq_len, movie_num]
        →logits:[batch*seq_len[mask not zero], movie_num] 为了方便计算loss
        '''

        embeds = self.embedding(x)
        # print(embeds)
        pack = nn_utils.rnn.pack_padded_sequence(embeds, seq_length, enforce_sorted=False, batch_first=True)
        out, h_n = self.gru(pack, None)
        unpacked, bz = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc1(unpacked)
        ranks = logits.view(-1, movie_num)[mask.view(-1) == 1.0]
        pred_tag = torch.argmax(logits, dim=-1)
        pred_tag = pred_tag.view(-1)[mask.view(-1) == 1.0]
        if loss_type == 'CrossEntropy':
            logits = logits.view(-1, movie_num)[mask.view(-1) == 1.0]
            loss = self.loss(logits, labels.view(-1))
        else:
            logits = self.final_activation(logits)
            ranks = logits.view(-1, movie_num)[mask.view(-1) == 1.0]
            loss = self.loss(logits[:,min(batch_size, min(seq_length))])

        return loss, pred_tag, ranks


load_data()
prepare_data()

movie_num = 3952
embedding_dim = 256
hidden_dim = 128
layer_dim = 4
output_dim = 3952
batch_size = 5
if args.train_model == 'gru':
    naiveRNN = GRU(movie_num, embedding_dim, hidden_dim, layer_dim, output_dim)
else:
    naiveRNN = RNN(movie_num, embedding_dim, hidden_dim, layer_dim, output_dim)

print(naiveRNN)

if torch.cuda.is_available():
    naiveRNN.cuda()   # 将所有的模型参数移动到GPU上


optimizer = torch.optim.RMSprop(naiveRNN.parameters(), lr=0.005)

train_loss_all = []
train_acc_all = []
train_acc_one_all = []
test_loss_all = []
test_acc_all = []
test_acc_one_all = []
num_epochs = 5
epoch_list = [i for i in range(num_epochs)]


def top_k(target, out):
    _, rank = torch.topk(out, 20, dim=-1, largest=True, sorted=True, out=None)
    target = target.view(-1, 1).expand_as(rank)
    hits = (target == rank).nonzero()
    if hits.size(0) == 0:
        return 0
    else:
        return float(hits.size(0)) / (target.size(0))

def train():
    global acc, acc_total, loss_num
    # print(len(train_datas))
    for epoch in range(num_epochs):
        acc_one = 0
        acc = 0
        acc_total = 0
        acc_one_total = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        naiveRNN.train()
        train_num = 0
        loss_num = 0
        # print(type(train_datas))
        train_features = Moviedata(train_data, train_label)
        train_dataloader = DataLoader(train_features, batch_size=batch_size,
                                      collate_fn=collate_fn, shuffle=True, drop_last=True, )
        for step, batch in enumerate(train_dataloader):
            input_ids, labels, labels_flatten, lengths, input_mask = batch

            input_ids = torch.tensor(input_ids).to(device)
            labels_flatten = torch.tensor(labels_flatten).to(device)
            input_mask = torch.tensor(input_mask)

            # print(labels_flatten.size())
            loss, pred, ranks = naiveRNN(input_ids, labels_flatten, lengths, input_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''if step % 500 == 0:
                print('step: ', step)
                print('this batch\'s loss is: ', loss.item())'''

            acc_one += len([item for item in range(len(pred)) if pred[item] == labels_flatten[item]])
            acc += top_k(labels_flatten, ranks)
            acc_one_total += len(pred)
            acc_total += 1
            loss_num += loss.item() * batch_size
            train_num += batch_size
            # if step % 500 == 0:
                # top_k(labels_flatten, ranks)
            if step % 500 == 0:
                print('step: ', step)
                print('loss is: ', loss_num / train_num)
                print('acc is: ', acc / acc_total)
                print('acc_one is: ', acc_one / acc_one_total)

        # model_name = os.path.join(args.model_path, "model_{0:05d}.pt".format(epoch))
        # torch.save(naiveRNN.state_dict(), model_name)
        # print("Save model as %s" % model_name)
        # print('epoch: ', epoch)
        print('loss is: ', loss_num / train_num)
        print('acc is: ', acc / acc_total)
        print('acc_one is: ', acc_one / acc_one_total)
        train_acc_one_all.append(acc_one / acc_one_total)
        train_acc_all.append(acc / acc_total)
        train_loss_all.append(loss_num / train_num)
        print('{} train loss: {:.4f}'.format(epoch, train_loss_all[-1]))
        print('{} train acc: {:.4f}'.format(epoch, train_acc_all[-1]))
        torch.save(naiveRNN.state_dict(), '{}\\{}'.format(args.model_path, epoch))
        evaluate()
    test()


def evaluate():
    test_total = 0
    test_total_one = 0
    test_loss_num = 0
    print('EXAMINATION TIME!!!')
    naiveRNN.eval()
    test_acc = 0
    test_acc_one = 0
    test_num = 0
    # print(type(train_datas))
    valid_features = Moviedata(valid_data, valid_label)
    valid_dataloader = DataLoader(valid_features, batch_size=batch_size,
                                      collate_fn=collate_fn, shuffle=True, drop_last=True, )
    for step, batch in enumerate(valid_dataloader):
        # print('step ', step)
        input_ids, labels, labels_flatten, lengths, input_mask = batch

        input_ids = torch.tensor(input_ids).to(device)
        # labels = torch.tensor(labels).to(device)
        labels_flatten = torch.tensor(labels_flatten).to(device)
        input_mask = torch.tensor(input_mask)

        # print(input_ids.size())

        loss, pred, ranks = naiveRNN(input_ids, labels_flatten, lengths, input_mask)


        test_acc += top_k(labels_flatten, ranks)
        test_acc_one += len([item for item in range(len(pred)) if pred[item] == labels_flatten[item]])
        test_total += 1
        test_total_one += len(pred)
        test_loss_num += loss.item() * batch_size
        test_num += batch_size
        if step % 500 == 0:
            print('step: ', step)
            print('loss is: ', test_loss_num / test_num)
            print('acc is: ', test_acc / test_total)

    print('loss is: ', test_loss_num / test_num)
    print('acc is: ', test_acc / test_total)
    print('acc_one is: ', test_acc_one / test_total_one)
    test_acc_all.append(test_acc / test_total)
    test_acc_one_all.append(test_acc_one / test_total_one)
    test_loss_all.append(test_loss_num / test_num)
    print('test loss: {:.4f}'.format(test_loss_all[-1]))
    print('test acc: {:.4f}'.format(test_acc_all[-1]))

def test():
    test_total = 0
    test_total_one = 0
    test_loss_num = 0
    print('------FINAL TEST------')
    naiveRNN.eval()
    test_acc = 0
    test_acc_one = 0
    test_num = 0
    # print(type(train_datas))
    test_features = Moviedata(test_data, test_label)
    test_dataloader = DataLoader(test_features, batch_size=batch_size,
                                      collate_fn=collate_fn, shuffle=True, drop_last=True, )
    for step, batch in enumerate(test_dataloader):
        # print('step ', step)
        input_ids, labels, labels_flatten, lengths, input_mask = batch

        input_ids = torch.tensor(input_ids).to(device)
        # labels = torch.tensor(labels).to(device)
        labels_flatten = torch.tensor(labels_flatten).to(device)
        input_mask = torch.tensor(input_mask)

        # print(input_ids.size())

        loss, pred, ranks = naiveRNN(input_ids, labels_flatten, lengths, input_mask)


        test_acc += top_k(labels_flatten, ranks)
        test_acc_one += len([item for item in range(len(pred)) if pred[item] == labels_flatten[item]])
        # temp_ans = top_k(labels, ranks, input_mask)
        # test_acc += temp_ans
        # test_acc -= temp_ans
        # test_acc += len([item for item in range(len(pred)) if pred[item] == labels_flatten[item]])
        test_total += 1
        test_total_one += len(pred)
        test_loss_num += loss.item() * batch_size
        test_num += batch_size
        if step % 500 == 0:
            print('step: ', step)
            print('loss is: ', test_loss_num / test_num)
            print('acc is: ', test_acc / test_total)

    print('loss is: ', test_loss_num / test_num)
    print('acc is: ', test_acc / test_total)
    print('acc_one is: ', test_acc_one / test_total_one)
    '''test_acc_all.append(test_acc / test_total)
    test_acc_one_all.append(test_acc_one / test_total_one)
    test_loss_all.append(test_loss_num / test_num)
    print('test loss: {:.4f}'.format(test_loss_all[-1]))
    print('test acc: {:.4f}'.format(test_acc_all[-1]))'''


def draw_figure():
    plt.figure(1)
    plt.title('{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.xlabel('num_epochs')
    plt.ylabel('loss')
    plt.xticks([i for i in range(num_epochs)])
    plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
    plt.plot(epoch_list, train_loss_all, label='train_loss')
    plt.plot(epoch_list, test_loss_all, label='test_loss')
    plt.legend(loc='upper right')
    plt.savefig('D:\\dataproject\\{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}_loss.png'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.show()
    plt.close()

    plt.figure(1)
    plt.title('{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.xlabel('num_epochs')
    plt.ylabel('acc')
    plt.xticks([i for i in range(num_epochs)])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.plot(epoch_list, train_acc_all, label='train_acc_topk')
    plt.plot(epoch_list, train_acc_one_all, label='train_acc_top1')
    plt.plot(epoch_list, test_acc_all, label='test_acc_topk')
    plt.plot(epoch_list, test_acc_one_all, label='test_acc_top1')
    plt.legend(loc='lower right')
    plt.savefig('D:\\dataproject\\{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}_acc.png'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.show()
    plt.close()

    plt.figure(1)
    plt.xlabel('num_epochs')
    plt.ylabel('loss \ acc')
    plt.xticks([i for i in range(num_epochs)])
    plt.title('{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.plot(epoch_list, train_loss_all, label='train_loss')
    plt.plot(epoch_list, test_loss_all, label='test_loss')
    plt.plot(epoch_list, train_acc_all, label='train_acc_topk')
    plt.plot(epoch_list, train_acc_one_all, label='train_acc_top1')
    plt.plot(epoch_list, test_acc_all, label='test_acc_topk')
    plt.plot(epoch_list, test_acc_one_all, label='test_acc_top1')
    plt.legend(loc='upper right')
    plt.xticks([i for i in range(num_epochs)])
    plt.savefig('D:\\dataproject\\{session_mode}_{layer_dim}_{loss_type}_{model_type}_{lr}_loss_acc.png'.format(
        session_mode=session_mode, layer_dim=layer_dim,
        loss_type=args.loss_type, model_type=args.train_model, lr=args.learn_rate))
    plt.show()
    plt.close()


train()
draw_figure()


