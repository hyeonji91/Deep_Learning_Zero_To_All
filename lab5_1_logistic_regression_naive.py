import torch


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs):
    # hypothesis = 1/(1+torch.exp(-(x_train.matmul(W)+b)))
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    # cost = -torch.mean((y_train*torch.log(hypothesis))+(1-y_train)*torch.log(1-hypothesis))
    cost = torch.nn.functional.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %100 == 0:
        print('Epoch: {}/{} Cost : {:.6f} W:{}, b:{:.5f}'.format(epoch+1, nb_epochs, cost.item(), W.flatten(), b.item()))


print(hypothesis)
prediction = hypothesis>=torch.FloatTensor([0.5])
print(prediction)
correct = prediction == y_train
print(correct)
correct = correct.type(torch.float64)
print("accuracy : {:2.2f}".format(torch.mean(correct)))