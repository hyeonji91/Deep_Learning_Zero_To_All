import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [5], [7]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr = 0.15)

nb_epochs = 20
for epoch in range(nb_epochs):
    hypothesis = x_train*W+b

    cost= torch.mean((hypothesis-y_train)**2)
    print("Epoch: {:4d}/{} W: {:.3f} b : {:.3f} Cost: {:.6f}".format(epoch, nb_epochs, W.item(), b.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
