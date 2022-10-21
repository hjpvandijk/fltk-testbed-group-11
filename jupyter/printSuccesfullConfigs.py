import pickle
import os
from sqlite3 import complete_statement

important = ["batch_size", "test_batch_size", "model", "Cpu", "Memory", "WORLD_SIZE"]
experiments = {}
for filename in os.listdir("configperjob"):
    f = open("configperjob/" + filename)
    id = filename[:filename.find(".txt")]
    experiments[id] = {}
    for i, line in enumerate(f):
        if i>8:
            splitted = line.split(": ")
            if(len(splitted)==2):
                feature = splitted[0].split()[0]
                if feature in important:
                    value = splitted[1].split()[0]
                    
                    experiments[id][feature] = value

# f = open('../configs/distributed_tasks/example_arrival_config.json')
# data = json.load(f)
# f.close()
# allConfigs = {}
# for i in data["trainTasks"]:
#     print(i)
#     configL = [data["hyperParameters"]["batchSize"], data["hyperParameters"]["testBatchSize"], data["networkConfiguration"]["network"],
#                data["resources"]["limits"]["cpu"], data["resources"]["limits"]["memory"], data["Paralell"]]
#     print(configL)
#     allConfigs[i] = configL
#     break

completedexperiments = []

objects = []
with (open("all_experiments.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
i=0
for obj in objects:
        for k, v in obj.items():
            if v.index.size > 3:
                id = k[k.find("trainjob-")+len("trainjob-"):k.find("-master")]
                completedexperiments.append(experiments[id])
                # print(v.loc[4.0].to_dict())
                i += 1

def get_batch_size(experiment):
    return experiment["batch_size"]

# print(get_batch_size(completedexperiments[0]))

sorted_list = sorted(completedexperiments, key=lambda t: (t["batch_size"], t["test_batch_size"],  t["WORLD_SIZE"]), reverse=True)


completedexperiments.sort(key=get_batch_size)
for exp in sorted_list:
    print(exp)     
            
print(len(completedexperiments))