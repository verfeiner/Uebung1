import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import pickle

# 定义加载 CIFAR-10 数据集的函数
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        cifar_batch = pickle.load(f, encoding='bytes')
        images = cifar_batch[b'data']
        labels = cifar_batch[b'labels']
        return images, labels

def load_all_cifar10_batches(base_folder, num_batches=2):
    all_images = []
    all_labels = []
    for i in range(1, num_batches + 1):
        batch_path = f"{base_folder}/data_batch_{i}"
        images, labels = load_cifar10_batch(batch_path)
        all_images.append(images)
        all_labels += labels

    return np.concatenate(all_images), np.array(all_labels)

# 指定 CIFAR-10 数据集所在的文件夹路径
cifar_folder = 'C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/cifar-10-batches-py'

# 加载所有 CIFAR-10 数据集批次
images, labels = load_all_cifar10_batches(cifar_folder)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.7, random_state=2)
#标准化
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.fit_transform(X_test)

# 初始化 KNN 分类器
knn_classifier = KNeighborsClassifier(n_neighbors=9)#每次判断一个未知的样本点时，就在该样本点附近找K个最近的点进行投票，这就是KNN中K的意义，通常K是不大于20的整数。

# 训练 KNN 分类器
knn_classifier.fit(X_train, y_train)#我们会选取K值在较小的范围，同时在验证集上准确率最高的那一个确定为最终的算法超参数K

# 在测试集上进行预测
y_pred = knn_classifier.predict(X_test)#为了克服降低样本不平衡对预测准确度的影响，我们可以对类别进行加权，例如对样本数量多的类别用较小的权重，而对样本数量少的类别，我们使用较大的权重。 另外，作为KNN算法唯一的一个超参数K,它的设定也会算法产生重要影响。因此，为了降低K值设定的影响，可以对距离加权。为每个点的距离增加一个权重，使得距离近的点可以得到更大的权重。

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
