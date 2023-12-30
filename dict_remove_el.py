# declare a dictionary
dict1 = {1: "one", 2: "two", 3: "three", 4: "four"}
print("Original Dictionary: ", dict1)
# using pop()
valDel = dict1.pop(1)
print("Dictionary after using pop(): ", dict1)
print("The value that was removed is: ", valDel)
# using pop()
valDel = dict1.pop(3)
print("Dictionary after using pop(): ", dict1)
print("The value that was removed is: ", valDel)
# using pop()
valDel = dict1.pop(3, "No Key Found")
print("Dictionary after using pop(): ", dict1)
print("The value that was removed is: ", valDel)
# using pop()
# if default value not given, error will be raised
print("Error raised: ")
valDel = dict1.pop(3)
