import numpy as np

np.random.seed(0)
values = np.random.randint(1, 10, size=10)
output = np.empty(len(values))
for i in range(len(values)):
    output[i] = values[i] ** 2

output_fast = values ** 2  # numpy vectorization

7 // 3  # floor division
7 / 3

array = np.arange(10).reshape(2, 5)
# below 2 are difference output type.
# numpy function
sum(array)
np.sum(array)
np.sum(array, axis=0)  # same whth sum(arrary)

ran_array = np.random.randint(1, 10, 10)
ran_array[np.argmax(ran_array)]

np.prod(ran_array)

prod = ran_array[0]
for i in range(len(ran_array) - 1):
    prod *= ran_array[i + 1]
prod

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n = np.array([1, 2, 3])

s = m + n
# masking
x = np.random.randint(10, size=(3, 4))
# x[x<5 & x>2] error!!
x[(x < 5) & (x > 2)]
x[x < 5]

# fancy indexing
x = np.arange(10) ** 2
x[[4, 7, 9]]

x = np.random.randint(100, size=(3, 5))
row = [0, 1, 2]
col = [2, 3, 1]

x[row, col]  # it returns x[0,2], x[1,3], x[2,1] why?

x[1, [1, 2]]  # guess
# x[1,[2,5]]  # out of range

# np.sort
x = np.random.randint(10, size=[3, 4])
x
x.sort()  # defaul axis=1
x
x.sort(axis=1)
x
x.sort(axis=0)
x
