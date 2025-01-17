import torch
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

# Update main.py
if __name__ == "__main__":
    from A.task_a import Task_a

    # Similar to above, Task B gets loaded similarly but separate scripts handle Task B.
    print("Starting training for Task A")
    Task_a(num_epochs=200, batch_size=32, lr=0.0001)

    # from B.task_b import Task_b
    # print("Starting training for Task B")
    # Task_b(num_epochs=10, batch_size=128, lr=0.001)
