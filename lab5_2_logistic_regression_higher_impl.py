import torch
import torch.nn.functional as F
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data= [[0],[0],[0],[1],[1],[1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# W = torch.zeros((2,1), requires_grad = True)
# b = torch.zeros(1, requires_grad = True)

model = BinaryClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs):
    # hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%100 == 0:
        print('Epoch: {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

#예측
hypothesis = model(x_train)
print(hypothesis)
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
#맞는지 확인
correct = prediction == y_train
print("correct: {}".format(correct))

#정확도
correct = correct.type(torch.float64)
accuracy = torch.mean(correct)
print("정확도: {}".format(accuracy.item()))