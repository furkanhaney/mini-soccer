import os

files = [f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".py")]
files.remove("dev.py")
files.sort()

with open("prompt.txt", "w") as f:
    for file in files:
        f.write(f"# Path: {file}\n")
        with open(file, "r") as g:
            f.write(g.read())
        f.write("\n")
