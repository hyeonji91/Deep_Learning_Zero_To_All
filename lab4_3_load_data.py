import torch
from torch.utils.data import Dataset
import torch.nn as nn


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
) # 2개씩 섞어서 나옴 = minibatch

#optimizer
model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    for batch_idx, samples in enumerate(dataloader):
        print(samples)
        x_train, y_train = samples

        prediction = model(x_train)

        cost = torch.nn.functional.mse_loss(prediction, y_train)

        #model개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('epoch : {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx, len(dataloader), cost.item(), len(dataloader)
        ))