lst1 = range(3, 5)
lst2 = range(5, 7)


def _print(a):
    x, y = a
    print(x, y)


print(len(list(map(_print, list(zip(lst1, lst2))))))
