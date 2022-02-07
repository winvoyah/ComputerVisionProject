# import numpy, matplotlib.pyplot and PIL's Image libs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print('open image file using Image open function in pillow library')
picture = Image.open("../data/bird.jpg")

print('cast the image data into numpy array')
# noinspection PyTypeChecker
picture_arr = np.array(picture)
print(picture_arr.shape)

print('use plt image show function to show the image')
plt.imshow(picture_arr)
plt.show()

print('copy picture array into a new variable so that we can adjust the RGB channel later on.')
arr_mod = picture_arr
print(arr_mod.shape)
print('zero out green and blue channels\' value and only keep color value in Red channel.')
arr_mod[:, :, 1] = 0  # remove colour blue
arr_mod[:, :, 2] = 0  # remove colour green
print('try to debug the pic_red variable and see what are the RGB value examples in 3 channels after zero out.')

print('show the result using imshow function')
print("\ncolours Blue and green removed\n")
plt.imshow(arr_mod)
plt.show()


print("save image to  directory")
result = Image.fromarray(arr_mod.astype(np.uint8))
result.save('../data/bird_only_red.jpeg')
