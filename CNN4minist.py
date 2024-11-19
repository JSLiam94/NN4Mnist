import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

# Super parameters ------------------------------------------------------------------------------------
batch_size = 128
learning_rate = 0.01
momentum = 0.5
EPOCH = 100
patience = 10  # 早停的 patience 参数，表示如果10个 epoch 内没有改善，就停止训练

# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Design model using class ----------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc(x)
        if loss_NA == 'NLLLoss':
            x = F.log_softmax(x, dim=1)  # NLLLoss使用log_softmax激活输出
        return x

# Construct loss and optimizer ----------------------------------------------------------------------------
model = Net()

loss_NA = 'CrossEntropyLoss'
if loss_NA == 'SmoothL1Loss':
    criterion = torch.nn.SmoothL1Loss()
elif loss_NA == 'CrossEntropyLoss':
    criterion = torch.nn.CrossEntropyLoss()
elif loss_NA == 'NLLLoss':
    criterion = torch.nn.NLLLoss()
else:
    print('loss function error')

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # Adam优化器
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # AdamW优化器

# 选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train and Test function ----------------------------------------------------------------------------
def train(epoch, best_acc, counter):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 移动数据到GPU

        optimizer.zero_grad()
        outputs = model(inputs)

        # 确保loss始终定义
        if loss_NA == 'SmoothL1Loss':
            target_one_hot = torch.zeros(target.size(0), 10, device=target.device).scatter_(1, target.view(-1, 1), 1)
            loss = criterion(outputs, target_one_hot)
        elif loss_NA == 'CrossEntropyLoss':
            loss = criterion(outputs, target)
        elif loss_NA == 'NLLLoss':
            loss = criterion(outputs, target)
        else:
            # 添加一个默认情况
            print(f"Unknown loss function: {loss_NA}. Using CrossEntropyLoss by default.")
            loss = torch.nn.CrossEntropyLoss()(outputs, target)  # 默认使用交叉熵损失

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 每300步输出一次
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0

    # 在每个epoch结束后进行验证，判断是否早停
    acc_test, _, _, _ = test(epoch)
    if acc_test > best_acc:
        best_acc = acc_test
        counter = 0  # 如果有提升，重置counter
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        return best_acc, counter, True  # 提前终止训练

    return best_acc, counter, False

def test(epoch):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print('[%d / %d]: Accuracy on test set: %.4f %%, Precision: %.4f, Recall: %.4f, F1: %.4f'
          % (epoch+1, EPOCH, 100 * acc, precision, recall, f1))
    return acc, precision, recall, f1

# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    best_acc = 0.0  # 最佳准确率初始化为0
    counter = 0  # 初始化计数器

    acc_list_test = []
    precision_list_test = []
    recall_list_test = []
    f1_list_test = []

    for epoch in range(EPOCH):
        best_acc, counter, early_stop = train(epoch, best_acc, counter)
        if early_stop:
            break  # 如果触发了早停，退出训练
        acc_test, precision_test, recall_test, f1_test = test(epoch)
        acc_list_test.append(acc_test)
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)
        f1_list_test.append(f1_test)

    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')

    # Plot precision
    plt.subplot(1, 4, 2)
    plt.plot(precision_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Precision On TestSet')

    # Plot recall
    plt.subplot(1, 4, 3)
    plt.plot(recall_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Recall On TestSet')

    # Plot f1
    plt.subplot(1, 4, 4)
    plt.plot(f1_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('F1 On TestSet')

    plt.tight_layout()
    plt.show()
