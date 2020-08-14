with open("滕王阁序.txt", "r") as f:
    content = f.read()

content = [w for w in content if w not in ["，", "。", "；", "\n", "?"]]

print(len(content))
print(len(set(content)))

with open("word_teng.txt", "w") as g:
    for c in content:
        g.write(c + "\n")