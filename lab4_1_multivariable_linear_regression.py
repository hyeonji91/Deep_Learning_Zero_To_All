import torch

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


optimizer = torch.optim.SGD([W, b], lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs):
    hypothesis = x_train.matmul(W)+b

    cost = torch.mean((hypothesis-y_train)**2)
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(),
        cost.item()
    ))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
