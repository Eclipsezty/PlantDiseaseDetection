

with open('E:/IDE/Anaconda/Scripts/requirements2.txt', 'r') as f:
    lines = f.readlines()

with open('E:/IDE/Anaconda/Scripts/requirements3.txt', 'w') as f:
    for line in lines:
        f.write('- ' + line)
print("ok")