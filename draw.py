import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model

# 构建CNN模型
model = Sequential()

# 输入层，假设输入为28x28的灰度图像
model.add(Conv2D(10, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

# 第二个卷积层
model.add(Conv2D(20, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 展平层，准备进入全连接层
model.add(Flatten())

# 第一个全连接层
model.add(Dense(320, activation='relu'))

# 输出层
model.add(Dense(10, activation='softmax'))

# 绘制模型结构图
# 调整图片大小和分辨率
plot_model(model, 
           to_file='cnn_model.png', 
           show_shapes=True, 
           show_layer_names=True, 
           dpi=150) 
