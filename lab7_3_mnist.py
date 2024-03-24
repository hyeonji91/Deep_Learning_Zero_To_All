import torch
from torch.utils.data import Dataset
import torchvision.datasets as dsets
from torchvision.transforms import transforms
import random
import matplotlib.pyplot as plt


#텐서 연산 시에 모든 텐서가 동일한 operation(전부 cpu나 전부 gpu)을 사용해야 합니다.
device = 'cpu' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root="MNIST_data/", train=True, download=True, transform=transforms.ToTensor())
mnist_test = dsets.MNIST(root="MNIST_data/", train=False, download=True, transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train, batch_size=batch_size, shuffle=True
)

linear = torch.nn.Linear(784, 10, bias=True).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    for x,y in data_loader:
        x = x.view(-1,28*28).to(device)
        y = y.to(device)

        hypothesis = linear(x)
        cost = criterion(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch

    print('Epoch: {}/{}, Cost = {:.6f} '.format(epoch, training_epochs, avg_cost))


with torch.no_grad():
    x_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = linear(x_test)
    correct_prediction=torch.argmax(prediction, dim=1).eq(y_test)

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()