import os
import shutil

# 获取train文件夹下所有文件名列表
train_filenames = os.listdir('train')
# 筛选出文件名以'cat'开头的文件列表（生成器形式）
train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
# 筛选出文件名以'dog'开头的文件列表（生成器形式）
train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)


def rmrf_mkdir(dirname):
    """
    函数功能：如果目录存在则删除（包括目录下所有内容），然后重新创建该目录
    参数：
    dirname：要操作的目录名称
    """
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


# 创建train3目录，如果已存在则先删除再创建
rmrf_mkdir('train3')
# 在train3目录下创建cat子目录
os.mkdir('train3/cat')
# 在train3目录下创建dog子目录
os.mkdir('train3/dog')

# 遍历文件名以'cat'开头的文件列表
for filename in train_cat:
    # 获取源文件的完整路径（在train目录下）
    src_path = os.path.join('train', filename)
    # 获取目标文件的完整路径（在train3/cat目录下）
    dst_path = os.path.join('train3/cat', filename)
    # 使用shutil.copy2函数将文件从源路径复制到目标路径，该函数会保留文件的元数据（如时间戳等）
    shutil.copy2(src_path, dst_path)

# 遍历文件名以'dog'开头的文件列表
for filename in train_dog:
    # 获取源文件的完整路径（在train目录下）
    src_path = os.path.join('train', filename)
    # 获取目标文件的完整路径（在train3/dog目录下）
    dst_path = os.path.join('train3/dog', filename)
    # 使用shutil.copy2函数将文件从源路径复制到目标路径，该函数会保留文件的元数据（如时间戳等）
    shutil.copy2(src_path, dst_path)