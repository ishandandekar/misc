from functools import reduce

numbers = list(range(1, 11, 1))

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

doubles = list(map(lambda x: x * 2, numbers))
print(doubles)

sum = reduce(lambda x, y: x + y, doubles)
print(sum)
