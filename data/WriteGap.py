from keras.models import *
from keras.layers import *
from keras.applications import ResNet50, InceptionV3, Xception
from keras.preprocessing.image import ImageDataGenerator
import h5py
import numpy as np
import os

# 定义函数write_gap，用于提取给定模型在特定数据集上的特征，并保存相关数据到HDF5文件
def write_gap(MODEL, image_size, lambda_func=None):
    """
    :param MODEL: 传入的预训练模型类，例如ResNet50、InceptionV3、Xception等
    :param image_size: 图像的尺寸，格式为(height, width)
    :param lambda_func: 可选的预处理函数，用于对输入图像进行额外的处理（如归一化等），默认为None
    """
    width = image_size[0]
    height = image_size[1]
    # 定义输入张量，形状为(height, width, 3)，对应图像的高度、宽度和通道数（RGB为3通道）
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        # 如果传入了预处理函数lambda_func，则对输入张量应用该函数进行处理
        x = Lambda(lambda_func)(x)

    # 创建基础模型，加载在ImageNet数据集上预训练的权重，不包含顶层（全连接层），用于提取特征
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    # 创建最终的模型，将基础模型的输入作为最终模型的输入，将基础模型输出经过全局平均池化后的结果作为最终模型的输出
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    # 从名为"train2"的目录中创建训练数据生成器，按照指定图像尺寸读取图像，不打乱顺序，批次大小设为16
    train_generator = gen.flow_from_directory("train3", image_size, shuffle=False,
                                              batch_size=16)
    # 从名为"test2"的目录中创建测试数据生成器，按照指定图像尺寸读取图像，不打乱顺序，批次大小设为16，且不返回类别标签（因为这里只是提取特征）
    test_generator = gen.flow_from_directory("test2", image_size, shuffle=False,
                                             batch_size=16, class_mode=None)

    # 用于存储训练集特征的列表
    train_features = []
    # 遍历训练数据生成器，获取每个批次的数据进行预测，提取特征并添加到train_features列表中
    for inputs_batch in train_generator:
        features_batch = model.predict(inputs_batch[0])
        train_features.append(features_batch)
    # 将列表中的特征数据沿第一个维度进行拼接，得到完整的训练集特征矩阵
    train = np.concatenate(train_features, axis=0)

    # 用于存储测试集特征的列表
    test_features = []
    # 遍历测试数据生成器，获取每个批次的数据进行预测，提取特征并添加到test_features列表中
    for inputs_batch in test_generator:
        features_batch = model.predict(inputs_batch[0])
        test_features.append(features_batch)
    # 将列表中的特征数据沿第一个维度进行拼接，得到完整的测试集特征矩阵
    test = np.concatenate(test_features, axis=0)

    # 打开（创建）名为"gap_{MODEL.func_name}.h5"的HDF5文件，用于保存数据
    with h5py.File("gap_%s.h5" % MODEL.func_name, 'w') as h:
        # 在HDF5文件中创建名为"train"的数据集，将训练集特征数据保存进去
        h.create_dataset("train", data=train)
        # 在HDF5文件中创建名为"test"的数据集，将测试集特征数据保存进去
        h.create_dataset("test", data=test)
        # 在HDF5文件中创建名为"label"的数据集，将训练集的类别标签数据保存进去
        h.create_dataset("label", data=train_generator.classes)


# 调用write_gap函数，使用ResNet50模型，图像尺寸为(224, 224)，无额外预处理函数，提取特征并保存数据
write_gap(ResNet50, (224, 224))
# 调用write_gap函数，使用InceptionV3模型，图像尺寸为(299, 299)，传入对应的预处理函数inception_v3.preprocess_input，提取特征并保存数据
write_gap(InceptionV3, (299, 299), InceptionV3.preprocess_input)
# 调用write_gap函数，使用Xception模型，图像尺寸为(299, 299)，传入对应的预处理函数xception.preprocess_input，提取特征并保存数据
write_gap(Xception, (299, 299), Xception.preprocess_input)