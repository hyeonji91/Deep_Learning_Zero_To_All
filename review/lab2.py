import torch

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1)
print(W)
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs):

    hypothesis = W * x_train

    cost = torch.mean((hypothesis - y_train)**2)
    print('Epoch: {}/{} W: {:.3f} Cost: {:.6f}'.format( epoch, nb_epochs, W.item(), cost.item()))

    gradient = torch.mean((hypothesis-y_train)*x_train)*2
    W = W - lr*gradient
