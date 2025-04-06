import os


def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出所有图片文件（假设图片文件是 .jpg, .jpeg, .png 格式）
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 第一步：重命名为 ly + 数字 的形式
    for i, image_file in enumerate(image_files, start=1):
        # 获取文件扩展名
        file_extension = os.path.splitext(image_file)[1]

        # 新文件名
        temp_file_name = f'ly{i}{file_extension}'

        # 获取完整的旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, image_file)
        temp_file_path = os.path.join(folder_path, temp_file_name)

        # 重命名文件
        os.rename(old_file_path, temp_file_path)

    # 更新文件列表
    temp_files = os.listdir(folder_path)
    temp_image_files = [file for file in temp_files if
                        file.lower().startswith('ly') and file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 第二步：重命名为 s + 数字 的形式
    for i, temp_image_file in enumerate(temp_image_files, start=1):
        # 获取文件扩展名
        file_extension = os.path.splitext(temp_image_file)[1]

        # 新文件名
        new_file_name = f'c{i}{file_extension}'

        # 获取完整的旧文件路径和新文件路径
        temp_file_path = os.path.join(folder_path, temp_image_file)
        new_file_path = os.path.join(folder_path, new_file_name)

        # 重命名文件
        os.rename(temp_file_path, new_file_path)

    print(f"重命名了 {len(image_files)} 张图片")


# 使用示例
folder_path = r'D:\Users\luyang\Desktop\最终选定图片\0000000\content'  # 替换为你图片文件夹的路径
rename_images_in_folder(folder_path)


# 更改  50  folder_path  路径  以及   37 new_file_name  c或者s  即可
