from torch import nn
from torchsummary import summary

class CustomConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 5 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2,4))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((3,4))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2,5))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2,4))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same"
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((4,4))
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        logits = self.linear(x)

        predictions = self.softmax(logits)

        return predictions

if __name__ == "__main__":
    model = CustomConvNet()
    summary(model.cuda(), (1, 96, 1360))