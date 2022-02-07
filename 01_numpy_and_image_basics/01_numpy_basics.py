# import numpy

import numpy as np

print('creating list [0,1,2,3,4] in python and print the type of your list')
mylist=[0,1,2,3,4]
print(type(mylist))

print('cast your list into a numpy arrary and print the type of your array')
arr1 = np.array(mylist)
print("type of array: ",  type(arr1))
print("data type of array: ", arr1.dtype)

print('create a 3x3 array filled with zeros')
arr2 =np.zeros((3,3))
print("\n", arr2)

print('create a 2x4 array filled with ones')
arr3 = np.ones((2,4))
print("\n", arr3)
print('create a 3x4 array filled with tens')
arr4 = np.ones((3,4)) *10
print("\n", arr4)

print('create a 3x2 array filled with random numbers')
arr5 = np.random.randn(3,2)
print("\n", arr5)
print('create a 3x2 arrary filled with random integers')
arr6 = np.random.randint(low=1, high =1000, size=(3,2), dtype= int)
print("\n", arr6)

print('create an 9 elements array [1,2,3,4,5,6,7,8,9]')
print('Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop)')
#list7 =[1,2,3,4,5,6,7,8,9]
arr7 = np.arange(1,10)
print("\n", arr7)

print('reshape your array (1x9) into a 3x3 matrix')
arr7 = arr7.reshape(3,3)
print("\n", arr7)
print('print the shape of your matrix')
print(arr7.shape)

print('print the maximum number in your matrix')
print(arr7.max())

print('print the minimum number in your matrix')
print(arr7.min())

print('print the mean value in your matrix')
print(arr7.mean())

print('retrieve the element on first row, second column')
#print(arr7)
print("\n", arr7[0:1,1:2])

print('retrieve all the elements on second column')
print(arr7[ : , 1:2])

print('retrieve all the elements on first row')
print(arr7[0:1, : ])

print('retrieve all the elements on first two rows and last two columns')
print(arr7[0:2, -2:])