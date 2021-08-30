import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import engine


# 데이터 다운로드
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="./resource",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="./resource",
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



# 추론 처리 정의
def inference(structure, index):
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    x, y = test_data[index][0], test_data[index][1]
    with torch.no_grad():
        pred = structure(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


#######################################################################

# 신경망 구조 정의
# CNN의 경우 아직 수정이 필요
structure = engine.model.NeuralNetwork()
structure.load_state_dict(
    torch.load("./resource/FashionMNIST/parameter/model1.pth")
    )

# 오차/최적화 함수 정의
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(structure.parameters(), lr=1e-3)

# Inference 과정
structure.eval()
index = 10

inference(structure, index)



