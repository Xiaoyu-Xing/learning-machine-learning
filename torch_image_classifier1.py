# Train an image classifier with cifar10 dataset

import torch
# Contains dataset and data loaders for vision training
import torchvision
import torchvision.transforms as transforms
# Two must haves for ANN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Linear full connections
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    # Feed forward

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(loss_function, learning_rate, cycles, device, trainloader):
    criterion = loss_function()
    # Stochastic gradient descent optimizing
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(cycles):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            samples, labels = data
            # Transfer data to GPU
            samples, labels = samples.to(device), labels.to(device)
            # Zero the gradients from previous round,
            # otherwise new gradient will addon and cause wrong result
            optimizer.zero_grad()
            # Forward prop
            outputs = net(samples)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Back prop, grad calculated automatically
            loss.backward()
            # Apply error correction to W and B automatically
            optimizer.step()
            # Monitor the process
            # Convert a scalar torch vector to python number
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (_ + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(testloader):
    correct = 0
    total = 0
    # No gradient calculation is necessary during
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Transfer data to GPU
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # torch.max(x, 1) return the max element in each row of x tensor and the index
            # predicted is the index for the max value, which is the index for predicted class
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # tensor equality check elementwise then sum and convert the scalar to python number
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    # transform.compose() to serialize a sequece of transformes to image
    # ToTensor() to transform PIL image or numpy image to torch image
    # channel*hight*width with range[0., 1.]
    # Normalize() to normalize a tensor image with (mean, std) for n channels
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        net.to(device)
    else:
        device = "cpu"

    train(nn.CrossEntropyLoss, 0.001, 2, device, trainloader)
    test(testloader)
