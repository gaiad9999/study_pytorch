import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import engine


# 데이터 다운로드
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="resource",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="resource",
    train=False,
    download=False,
    transform=ToTensor(),
)


# 데이터 로드
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# GPU 동작 테스트
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



#######################################################################


# 신경망 구조 정의
structure = engine.model.NeuralNetwork().to(device)
print(structure)

# 오차/최적화 함수 정의
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(structure.parameters(), lr=1e-3)

# 학습
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    engine.learn.train(train_dataloader, structure, loss_fn, optimizer, device)
    engine.learn.test(test_dataloader, structure, loss_fn, device)
print("Done!")

# 학습결과 파라메타 저장
torch.save(structure.state_dict(),
           "./resource/FashionMNIST/parameter/model1.pth")
print("Saved PyTorch Model State to model.pth")


