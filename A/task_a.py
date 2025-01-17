from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image

import medmnist
from medmnist import INFO, Evaluator


def Task_a(num_epochs, batch_size, lr):

    data_flag = "breastmnist"
    # data_flag = "bloodmnist"
    download = True

    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    lr = lr

    info = INFO[data_flag]
    print(info)
    task = info["task"]
    print(task)
    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    # preprocessing
    data_transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            # ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # load the data
    train_dataset = DataClass(
        split="train", transform=data_transform, download=download
    )
    test_dataset = DataClass(split="test", transform=data_transform, download=download)

    pil_dataset = DataClass(split="train", download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_loader_at_eval = data.DataLoader(
        dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
    )

    # visualization
    train_dataset.montage(length=20).save("train_montage.png")

    model = Net(in_channels=n_channels, num_classes=n_classes)

    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Initialize lists to track loss and accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []

    # train

    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        train_loss = 0

        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Record training loss
        train_losses.append(train_loss / len(train_loader))

        # Validate the model
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)

                if task == "multi-label, binary-class":
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                    predicted = outputs > 0.5
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)

                val_loss += loss.item()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_losses.append(val_loss / len(test_loader))
        val_accuracies.append(100 * correct / total)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.2f}%"
        )

        # if val_accuracies[-1] >= 80.0 and optimizer.param_groups[0]["lr"] > 1e-4:
        #     new_lr = optimizer.param_groups[0]["lr"] * 0.1  # Reduce by a factor of 0.1
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = new_lr
        #     print(f"Updated Learning Rate: {new_lr}")

    # Plot the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("learning_curve_loss_task_A.png")

    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.savefig("learning_curve_accuracy_task_A.png")

    print("==> Evaluating ...")
    test("train", model, train_loader_at_eval, task, data_flag)
    test("test", model, test_loader, task, data_flag)


# evaluation
def test(split, model, data_lodaer, task, data_flag):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    data_loader = data_lodaer

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print("%s  auc: %.3f  acc:%.3f" % (split, *metrics))


class Net(nn.Module):
    """Define a simple CNN model."""

    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout(0.9),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.9),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # Testing:
    # def forward(self, x):
    #     print(f"Input size: {x.size()}")  # Print input size
    #     x = self.layer1(x)
    #     print(f"After layer1: {x.size()}")
    #     x = self.layer2(x)
    #     print(f"After layer2: {x.size()}")
    #     x = self.layer3(x)
    #     print(f"After layer3: {x.size()}")
    #     x = self.layer4(x)
    #     print(f"After layer4: {x.size()}")
    #     x = self.layer5(x)
    #     print(f"After layer5: {x.size()}")
    #     x = x.view(x.size(0), -1)  # Flatten for fully connected layers
    #     print(f"After flattening: {x.size()}")
    #     x = self.fc(x)
    #     print(f"After fully connected layers: {x.size()}")
    #     return x
