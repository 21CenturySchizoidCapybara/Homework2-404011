# MLP
import tensorflow as tf 
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Deformation
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# Normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# Split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.5, random_state = 42)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
# Define Model
model1 = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])
model2 = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])
model3 = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model4 = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,), kernel_regularizer=keras.regularizers.l1(0.01)),
    keras.layers.Dense(10, activation='softmax')
])
model5 = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,), kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(10, activation='softmax')
])
model6 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)), 
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# Compile model
model1.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model3.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model4.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model5.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model6.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Data Augmentation
data_augmentation = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train Model
history1 = model1.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    batch_size=32
)
history2 = model2.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    batch_size=32,
    callbacks=[early_stopping]
)
history3 = model3.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    batch_size=32
)
history4 = model4.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    batch_size=32,
)
history5 = model5.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    batch_size=32,
)
xv_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
xv_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
print(xv_train.shape, xv_val.shape)
history6 = model6.fit(
    data_augmentation.flow(xv_train, y_train, batch_size=32),  
    epochs=10,
    validation_data=(xv_val, y_val), 
    steps_per_epoch=len(xv_train) // 32  
)

# CNN
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.feature_size = self._get_conv_output((3, 32, 32))

        self.fc1 = nn.Linear(self.feature_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.forward_features(input)
        return int(np.prod(output.size()))

    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / (i + 1))
    train_accuracies.append(100 * correct_train / total_train)
    
    net.eval()
    running_loss = 0.0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()
    
    valid_losses.append(running_loss / len(testloader))
    valid_accuracies.append(100 * correct_valid / total_valid)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.3f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Loss: {valid_losses[-1]:.3f}, Val Acc: {valid_accuracies[-1]:.2f}%')

net.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.2f}%')
