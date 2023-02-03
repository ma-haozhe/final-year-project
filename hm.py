filename = "100000000000000.jpg" 
print("file name:", filename, "\n")
if filename.endswith('.jpg'):
    count = 0
    for char in filename:
        print(char)
        if char == '0':
            count += 1
        else:
            continue
    print("0 count is ", count)
