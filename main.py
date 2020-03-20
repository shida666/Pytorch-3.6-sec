# 3.6
import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt  #加

print(torch.__version__)
print(torchvision.__version__)

# 3.6.1
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 3.6.2
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))

# 3.6.3
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

X = torch.rand((2, 5),requires_grad=True)
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# 3.6.4
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# 3.6.5
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 3.6.6
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

print(accuracy(y_hat, y))


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

print(evaluate_accuracy(test_iter, net))
print('3.6.6 over')

# 3.6.7
#num_epochs, lr = 5, 0.1  #源文件
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

#            x = Variable(x, requires_grad=True)  # 生成变量
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size, [W, b], lr)
print('3.6.7 over')


# 3.6.8
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

plt.show() #加

print('3.6.8 over')

'''
1.4.0
0.5.0
tensor([[5, 7, 9]])
tensor([[ 6],
        [15]])
tensor([[0.2041, 0.3145, 0.1532, 0.1868, 0.1415],
        [0.1032, 0.2025, 0.2038, 0.2379, 0.2527]], grad_fn=<DivBackward0>) tensor([1., 1.], grad_fn=<SumBackward1>)
0.5
0.0691
3.6.6 over
epoch 1, loss 1.0149, train acc 0.730, test acc 0.812
epoch 2, loss 0.6726, train acc 0.798, test acc 0.819
epoch 3, loss 0.6100, train acc 0.813, test acc 0.823
3.6.7 over
3.6.8 over

Process finished with exit code 0
'''