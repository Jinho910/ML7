a = [1, 2, 3, 4]
b = [5, 6, 7, 8]

c = list(map(lambda x, y: x + y, a, b))
print(c)

# filter even numbner
d = list(filter(lambda x: x % 2 == 0, a))
print(d)

from functools import reduce

e = reduce(lambda x, y: x + y, a)
print(e)

