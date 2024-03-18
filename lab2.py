import numpy as np
import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1)
# b = torch.zeros(1, requires_grad=True)

lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs):
    hypothesis = W * x_train

    cost = torch.mean((hypothesis - y_train)**2)
    gradient = torch.mean((W*x_train - y_train) * x_train)*2

    print('Epoch {:4d}/{} W:{:.3f}, Cost: {:6f}, g : {}'.format(epoch, nb_epochs, W.item(), cost.item(), gradient.item()))

    W -= lr * gradient

