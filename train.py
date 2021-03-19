import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class CNNClassifier(nn.Module):
    """
    Create a simple CNN model inherited from nn.Module
    """
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolution1 [100, 1, 28, 28] -> [100, 16, 26, 26]
        # MaxPool [100, 16, 26, 26] -> [100, 16, 13, 13]
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        # Convolution2 [100, 16, 13, 13] -> [100, 32, 11, 11]
        # MaxPool [100, 32, 11, 11] -> [100, 32, 5, 5]
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        # flatten [100, 32, 5, 5] -> [100, 800]
        x = x.view(-1, 800)
        # MLP [100, 800] -> [100, 100]
        x = self.relu(self.fc1(x))
        # MLP [100, 100] -> [100, 10]
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# define transform data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, ), std=(0.5, ))])

# init variable
_batch_size = 100
lr = 1e-3

# load data
train_data = MNIST("data", download=True, train=True, transform=transform)
train_data, valid_data = random_split(train_data, [50000, 10000])

print("Train: {}, Valid: {}".format(len(train_data), len(valid_data)))

# wrapped dataset into dataLoader for training
train_loader = DataLoader(dataset=train_data, batch_size=_batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=_batch_size, shuffle=False)

# create model
model = CNNClassifier()

# train
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):

    ### TRAIN MODE
    epoch_total_loss = 0
    model.train()
    for nth_batch, inputs in enumerate(train_loader):
        # get input data
        img_tensor = inputs[0]
        labels = inputs[1]

        # predict
        out = model(img_tensor)

        # calculate loss
        loss = criterion(out, labels)
        epoch_total_loss += loss

        # back propagation each batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate average loss every epoch
    avg_loss_train = epoch_total_loss / len(train_loader)

    ### EVALUATE MODE
    epoch_total_loss = 0
    model.eval()
    with torch.no_grad():
        for nth_batch, inputs in enumerate(valid_loader):
            # get input data
            img_tensor = inputs[0]
            labels = inputs[1]

            # predict
            out = model(img_tensor)

            # calculate loss
            loss = criterion(out, labels)
            epoch_total_loss += loss

    # calculate average loss every epoch
    avg_loss_valid = epoch_total_loss / len(valid_loader)

    # print training log every epoch
    log = "Epoch {} / {} || [*Train*] loss = {:.5f} [*Valid*] loss = {:.5f}".format(epoch + 1, 10, avg_loss_train, avg_loss_valid)
    print(log)

    # save model
    torch.save(model.state_dict(), 'model\\epoch_{}.pth'.format(epoch + 1))