import torch
import torch.nn.functional as F

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

w = torch.zeros((4,3), requires_grad=True)
d = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w, d], lr = 0.1)

nb_epochs = 1000
for epoch in range(nb_epochs):
    hypothesis = F.softmax(x_train.matmul(w)+d, dim=1)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    cost = (y_one_hot*-torch.log(hypothesis)).sum(dim = 1).mean()

    """F.log_softmax"""
    # z = x_train.matmul(w)+d
    # y_one_hot = torch.zeros_like(z)
    # y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    # cost = (y_one_hot*-F.log_softmax(hypothesis, dim=1)).sum(dim = 1).mean()

    """F.nll_loss"""
    # z = x_train.matmul(w) + d
    # cost = F.nll_loss(F.log_softmax(z, dim=1), y_train)

    """F.cross_entropy"""
    # z = x_train.matmul(w)+d
    # cost = F.cross_entropy(z,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch: {}/{} Cost :{:.6f}'.format(epoch, nb_epochs, cost))

