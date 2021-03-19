import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
# load data
test_data = MNIST("data", download=True, train=False, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

# load model
model_path = 'model\\epoch_10.pth'
model = CNNClassifier()
model.load_state_dict(torch.load(model_path))

# test
for nth_data, inputs in enumerate(test_loader):
    # get input data
    img_tensor = inputs[0]
    label = inputs[1]

    # predict
    out = model(img_tensor)
    out_ans = torch.argmax(out)

    # plot result
    plt.Figure()
    plt.title("gt={}, pd={}".format(int(label), int(int(out_ans))))
    plt.imshow(img_tensor.data.numpy()[0][0], 'gray')
    plt.show()
