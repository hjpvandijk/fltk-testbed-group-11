import pickle
import json

f = open('./example_arrival_config.json')
data = json.load(f)
f.close()
allConfigs = {}
for i in data:
    configL = [data[i]["batch_size"], data[i]["test_batch_size"], data[i]["model"],
               data[i]["resources"]["limits"]["cpu"], data[i]["resources"]["limits"]["memory"], data[i]["Paralell"]]
    allConfigs[i] = configL
objects = []
with (open("./all_experiments_full_filtered.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for key in allConfigs.keys():
    for obj in objects:
        for k, v in obj.items():
            if key in k:
                if v.index.size > 3:
                    allConfigs.update({key: (allConfigs[key], v.loc[4.0].to_dict())})
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []
g6 = []
y2 = []
y3 = []
y4 = []
for k, v in allConfigs.items():
    if type(v) == type((1, 2)):
        g1.append(v[0][0])
        g2.append(v[0][1])
        g3.append(v[0][2])
        g4.append(v[0][3])
        g5.append(v[0][4])
        g6.append(v[0][5])
        y2.append(v[1]["test accuracy per epoch"])
        y3.append(v[1]["test latency per epoch"])
        y4.append(v[1]["train latency per epoch"])
print(g1)
print(g2)
print(g3)
print(g4)
print(g5)
print(g6)
print(y2)
print(y3)
print(y4)
