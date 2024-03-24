import torch
from torch import nn


class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,3)

    def forward(self,x):
        return self.linear(x)

def train(model, optimizer, x_train, y_train):
    nb_epoch = 20
    for epoch in range(nb_epoch):
        z = model(x_train)
        cost = nn.functional.cross_entropy(z, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch: {}/{}, Cost: {}'.format(epoch, nb_epoch, cost))

def test (model, optimizer, x_test, y_test):
    prediction = model(x_test)
    prediction_classes = prediction.max(1)[1]
    correct_count = (prediction_classes == y_test).sum().item()
    cost = nn.functional.cross_entropy(prediction, y_test)
    print('Accuracy: {}% Cost : {:.6f}'.format(correct_count / len(y_test)*100, cost))

x_train = torch.FloatTensor([[1, 2, 1, ],
           [1,3,2],
           [1, 3, 4],
           [1, 5, 5],
           [1, 7, 5],
           [1, 2, 5],
           [1, 6, 6],
           [1, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
x_test = torch.FloatTensor([[2,1,1],[3,1,2],[3,3,4]])
y_test = torch.LongTensor([2,2,2])

model = SoftmaxClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)