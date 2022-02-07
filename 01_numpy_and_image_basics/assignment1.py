# import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Create and print a 3 by 3 array where every number is a 15
arr0 = np.ones((3, 3)) * 15
print(arr0)
# print out what are the largest and smalled values in the array below
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
min = arr.min
max = arr.max
print("min value is : %d and maximum value is : %d", min, max)

# import pyplot lib from matplotlib and Image lib from PIL
# import done at the start. Avoids pycharm error alert


# use PIL and matplotlib to read and display the ../data/zebra.jpg image
pic_zebra = Image.open("../data/zebra.jpg")
plt.imshow(pic_zebra)
plt.show()

# convert the image to a numpy array and print the shape of the array
# noinspection PyTypeChecker
pic_arr = np.array(pic_zebra)
print(pic_arr.shape)
print(pic_arr)
# use slicing to set the RED and GREEN channels of the picture to 0, then use imshow() to show the isolated blue channel
pic_arr1 = pic_arr
pic_arr1[:, :, 0] = 0  # red channel converted to 0
pic_arr1[:, :, 1] = 0  # green channel converted to 0

plt.imshow(pic_arr1)
plt.show()

print("save image to  directory")
result = Image.fromarray(pic_arr1.astype(np.uint8))
result.save('../data/zebra_only_blue.jpeg')
