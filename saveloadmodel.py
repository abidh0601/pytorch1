import torchvision.models as models
import torch.onnx as onnx
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()


learning_rate = 1e-3
batch_size = 64
epochs = 5

# Want to minimize loss function during training

loss_fn = nn.CrossEntropyLoss()

# Optimization: Adjust model paramters to reduce model error in each training step

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# train loop
#   optimizer.zero_grad() reset the gradients of model paramters. Gradients add up by default
#       Reset to zero prevents double counting
#   Back propagate the prediction loss with a call to loss.backwards
#   optimizer.step() adjust parameters by the gradients collected in the backward pass

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10

# for t in range(epochs):
#     print(f"Epoch {t+1}\n--------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")


# ---------------------------Saving and loading wegihts --------------------
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')


model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
# --------------------------------------------------------------------------

# --------------------------Saving and laoding with shapes------------------


torch.save(model, "model.pth")
model = torch.load('model.pth')
# ------------------------------------------------------------------------

# ----------------------Exporting  model to ONNX -------------------------

input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')
