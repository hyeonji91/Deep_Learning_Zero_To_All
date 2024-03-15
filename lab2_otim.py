import torch

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs+1):
    hypothesis = x_train * W

    cost = torch.mean((hypothesis - y_train)**2)
    print("Epoch: {:4d}/{} W: {:.3f} Cost: {:.6f}".format(epoch, nb_epochs, W.item(), cost.item()))

    optimizer.zero_grad() #초기화
    cost.backward() #gradient계산
    optimizer.step() #gradent descent