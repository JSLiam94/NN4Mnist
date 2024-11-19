import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 归一化像素值
train_images = train_images / 255.0
test_images = test_images / 255.0

# 设置损失函数类型
loss = 'MeanSquaredError'  # 可以选择 'SparseCategoricalCrossentropy', 'CategoricalCrossentropy', 'MeanSquaredError', 'HuberLoss' 等
optimizer = 'sgd'  # 优化器

# 根据损失函数类型选择标签处理方式和损失函数
if loss == 'SparseCategoricalCrossentropy':
    # 不需要转换标签，直接使用整数编码
    train_labels_processed = train_labels
    test_labels_processed = test_labels
    model_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
elif loss == 'CategoricalCrossentropy':
    # 需要将标签转换为one-hot编码
    train_labels_processed = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels_processed = tf.keras.utils.to_categorical(test_labels, 10)
    model_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


elif loss == 'MeanSquaredError':
    # 回归任务，标签为连续数值
    train_labels_processed = train_labels.astype(float)  
    test_labels_processed = test_labels.astype(float)
    model_loss = tf.keras.losses.MeanSquaredError()

elif loss == 'HuberLoss':
    # Huber损失，适用于回归任务
    train_labels_processed = train_labels.astype(float)  
    test_labels_processed = test_labels.astype(float)
    model_loss = tf.keras.losses.Huber()

# 构建模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 将输入展平
    layers.Dense(128, activation='relu'),  # 隐藏层
    layers.Dropout(0.2),  # Dropout 层
    layers.Dense(10)  # 输出层，10个神经元对应10个分类
])

# 编译模型
model.compile(optimizer=optimizer,
              loss=model_loss,  # 根据损失函数类型选择
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels_processed, epochs=10,
                    validation_data=(test_images, test_labels_processed))

# 绘制训练和验证的准确率和损失图
plt.figure(figsize=(12, 4))

# 准确率图
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失图
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels_processed, verbose=2)
print('\nTest accuracy:', test_acc)

# 预测新数据
predictions = model.predict(test_images)
print(predictions[:5])  # 显示前5个预测结果
