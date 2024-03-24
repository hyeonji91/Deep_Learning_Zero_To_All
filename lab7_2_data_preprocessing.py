import torch
from torch import nn


class MultivariateLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        prediction = model(x_train)
        cost = nn.functional.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch: {}/{}, Cost: {}'.format(epoch, nb_epochs, cost.item()))



x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#preprocessing
mu = x_train.mean(dim=0)
sigma = x_train.std(dim = 0)
norm_x_train = (x_train - mu)/sigma
print(norm_x_train)

model = MultivariateLinearRegression()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
train(model, optimizer, norm_x_train, y_train)
train(model, optimizer, x_train, y_train) #lr = 1e-5 learning rate을 조절하면 코스트가 줄긴 주는데