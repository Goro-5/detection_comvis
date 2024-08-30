import os

directory = "PS-18SU/BACK"
for root, dirs, files in os.walk(directory):
    print(root)
    print(dirs)
    print(files)
    print()