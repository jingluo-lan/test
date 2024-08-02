import os
import shutil

def organize_files_by_label(src_directory, dest_with_label, dest_without_label):
    """
    将指定目录下的文件按文件名是否包含"label"分到两个目录中。

    参数:
    src_directory (str): 源目录，包含所有待处理的文件。
    dest_with_label (str): 包含"label"的文件存放目录。
    dest_without_label (str): 不包含"label"的文件存放目录。
    """
    # 确保目标目录存在
    os.makedirs(dest_with_label, exist_ok=True)
    os.makedirs(dest_without_label, exist_ok=True)

    # 遍历源目录中的文件
    for filename in os.listdir(src_directory):
        if "label" in filename:
            dest = dest_with_label
        else:
            dest = dest_without_label

        # 构造完整的源文件路径和目标文件路径
        src_path = os.path.join(src_directory, filename)
        dest_path = os.path.join(dest, filename)

        # 移动文件到目标目录
        shutil.copy(src_path, dest_path)
        print(f"Copy {filename} to {dest}")

# 示例用法
src_directory = '/home/user/lanzheng/CycleGAN/UNet_40/data/mmwhs/mr/mr_train'  # 替换为你的源目录路径
dest_with_label = '/home/user/lanzheng/CycleGAN/CycleGAN_UNet/Combine_Cyc-UNet/datasets/train_mr_labels'  # 替换为包含"label"的文件存放目录
dest_without_label = '/home/user/lanzheng/CycleGAN/CycleGAN_UNet/Combine_Cyc-UNet/datasets/train_mr'  # 替换为不包含"label"的文件存放目录

organize_files_by_label(src_directory, dest_with_label, dest_without_label)
