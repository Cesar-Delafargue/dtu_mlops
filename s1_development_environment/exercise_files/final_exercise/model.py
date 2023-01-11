from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64,10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.classifier(x)    

