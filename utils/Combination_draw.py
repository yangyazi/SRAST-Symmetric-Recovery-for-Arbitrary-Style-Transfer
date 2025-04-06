import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 设定图片文件夹路径
image_folder = 'out'

# 加载图片
image1 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_1_20200429082107.jpg'))
image2 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_20200428220820 - 副本.jpg'))
image3 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_20200428220829 - 副本.jpg'))
image4 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_20200428220911.jpg'))
image5 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_antimonocromatismo.jpg'))
image6 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_asheville.jpg'))
image7 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_brushstrokes.jpg'))
image8 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_candy.jpg'))
image9 = mpimg.imread(os.path.join(image_folder, 'avril_stylized_contrast_of_forms.jpg'))

# 创建figure
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# 显示图片并添加标题
axs[0, 0].imshow(image1)
axs[0, 0].set_title('Image 1')
axs[0, 1].imshow(image2)
axs[0, 1].set_title('Image 2')
axs[0, 2].imshow(image3)
axs[0, 2].set_title('Image 3')
axs[1, 0].imshow(image4)
axs[1, 0].set_title('Image 4')
axs[1, 1].imshow(image5)
axs[1, 1].set_title('Image 5')
axs[1, 2].imshow(image6)
axs[1, 2].set_title('Image 6')
axs[2, 0].imshow(image7)
axs[2, 0].set_title('Image 7')
axs[2, 1].imshow(image8)
axs[2, 1].set_title('Image 8')
axs[2, 2].imshow(image9)
axs[2, 2].set_title('Image 9')

# 隐藏坐标轴
for ax in axs.flat:
    ax.axis('off')

# 调整布局
plt.tight_layout()
plt.show()
