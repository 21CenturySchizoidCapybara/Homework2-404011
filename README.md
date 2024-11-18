import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 变形
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# 归一
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 拆分
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.5, random_state = 42)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
# 定义Keras的Sequential API模型
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
# 编译模型
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
# 数据增强
data_augmentation = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
# 早停
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# 训练模型
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
# 重塑数据以适应ImageDataGenerator
xv_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
xv_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
print(xv_train.shape, xv_val.shape)
history6 = model6.fit(
    data_augmentation.flow(xv_train, y_train, batch_size=32),  
    epochs=10,
    validation_data=(xv_val, y_val), 
    steps_per_epoch=len(xv_train) // 32  
)
