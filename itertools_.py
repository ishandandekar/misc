import itertools

data = [100, 200, 300, 400]

# Counter
daily_data = list(zip(itertools.count(), data))
print(daily_data)

# Cycle
counter = itertools.cycle(["On", "Off"])
daily_data = list(zip(counter, data))
print(daily_data)

# Repeat
counter = itertools.repeat(2, 3)
squares = list(map(pow, range(10), itertools.repeat(2)))
print(squares)

# Starmap
squares = list(itertools.starmap(pow, [(0, 2), (1, 2), (2, 2)]))
print(squares)

# Combinations
letters = ["a", "b", "c", "d"]
numbers = [0, 1, 2, 3]
names = ["Corey", "Ishan"]
result = itertools.combinations(letters, 2)
for item in result:
    print(item)

result = itertools.product(numbers, repeat=4)
for item in result:
    print(item)

print("\n" * 3)
result = itertools.combinations_with_replacement(numbers, r=4)
for item in result:
    print(item)

# Chain
combined = itertools.chain(letters, numbers, names)
for item in combined:
    print(item)
