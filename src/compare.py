import csv

CSV1 = "results.csv"
CSV2 = "train_lables.csv"

f1 = open(CSV1, newline=''); f2 = open(CSV2, newline='')
reader1 = csv.reader(f1); reader2 = csv.reader(f2)
rows1 = list(reader1); rows2 = list(reader2)
correct = 0
for i in range(1,len(rows1)):
    correct += ([int(x.replace('.', '')) for x in rows1[i][-5:]] == [int(x.replace('.','')) for x in rows2[i][-5:]])

print(f"Accuracy: {correct/len(rows1):.4f}")
