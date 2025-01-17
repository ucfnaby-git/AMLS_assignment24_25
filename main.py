import torch
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


if __name__ == "__main__":
    from A.task_a import Task_a

    print("Starting training for Task A")
    Task_a(num_epochs=200, batch_size=32, lr=0.0001)

    from B.task_b import Task_b

    print("Starting training for Task B")
    Task_b(num_epochs=20, batch_size=256, lr=0.001)
