from torch.nn.modules.activation import Softmax
from torchvision import datasets, transforms
from torch import nn
import os
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# --------------------Download Dataset for training and testing-----------------------

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
    transform=ToTensor
)

# ---------------------------------------------------------------------------------------

# ----------------------------Visualizing Dataset with plt ------------------------------

# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# --------------------------------------------------------------------------------------------

# ---------------------------Preparing Data with DataLoaders----------------------------------


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# ----------------------------------------------------------------------------------------------

# -----------------------------------Iterate through DataLoader-------------------------------
# train_features, train_lables = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_lables.size()}")
# img = train_features[0].squeeze()
# label = train_lables[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Labal: {train_features[0]}")
# --------------------------------------------------------------------------------------------------

# -------------------------------------Build the Model --------------------------------------------


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


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


model = NeuralNetwork().to(device)
# print(model)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_porbab = nn.Softmax(dim=1)(logits)
# y_pred = pred_porbab.argmax(1)
# print(f"Predicted class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


print(f"Before ReL: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReL: {hidden1}\n\n")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

logits = seq_modules(input_image)

softmax = Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)


print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")
