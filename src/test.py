
import csv
import math

f = open("training/covid_training.tsv", "r", encoding="utf-8")
reader = csv.reader(f, delimiter="\t")

# We don't want the header
rows = list(reader)[1:]

classes = ["yes", "no"]
twt = [r[1] for r in rows]
factual = [r[2] for r in rows]

tweets = dict(zip(twt, factual))

vocab = {}

for t, f in tweets.items():
    s = list(t.split(" "))
    for word in s:
        word = word.strip(".,?!@#:\"“-—\'()").lower()
        if not word in vocab:
            vocab[word] = {classes[0] : int, classes[1] : int}
            for c in classes:
                vocab[word][c] = 0
        if f == classes[0]:
            vocab[word][classes[0]] += 1
        elif f == classes[1]:
            vocab[word][classes[1]] += 1


# for word in vocab:
#     v[word] = []
#     for c in classes:
#         v[word].append(0)

# print(vocab)

conditional_p = {str : float}

print(conditional_p)

smoothing = 0.01

def totalInClass(c):
    sum = 0
    for _, b in vocab.items():
        sum += b[c]
    
    return sum

total_in_class = {classes[0] : totalInClass(classes[0]), classes[1] : totalInClass(classes[1])}

for c in classes:
    for word in vocab:
        conditional_p[word] = math.log10((vocab[word][c] + smoothing)/total_in_class[c])

print(conditional_p)

p_c = {"yes": float, "no" : float}

for c in classes:
    p_c[c] = factual.count(c)/len(tweets)

# print(p_c)
# print(len(vocab), vocab)