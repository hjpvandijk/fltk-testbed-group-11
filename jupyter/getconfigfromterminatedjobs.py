import os
from pydoc import describe

# ret = os.popen("kubectl get configmap -n test --sort-by=.metadata.creationTimestamp --no-headers -o custom-columns=:.metadata.name").read()

jobs = os.popen("kubectl get pytorchjobs.kubeflow.org -n test -o custom-columns=NAME:.metadata.name").read()



splitted = jobs.split('\n')

isExist = os.path.exists("epochdata")
if not isExist:
    os.mkdir("epochdata")

isExist = os.path.exists("describepodsterminated")
if not isExist:
    os.mkdir("describepodsterminated")

foundids = []
nrran = 0

for i in splitted:
    if "trainjob" in i:
        id = i[len("trainjob-"):]
        if id not in foundids:
            foundids.append(id)
            print(i)
            nrran += 1

            describepod = os.popen("kubectl describe pytorchjobs.kubeflow.org " + i + " -n test").read()

            retconfig = os.popen("kubectl describe configmap -n test worker-" + id + "-0").read()
            resources = describepod.find("Resources")
            volumemounts = describepod.find("Volume Mounts")
            resourceslist = describepod[resources:volumemounts]

            worker_replicas = describepod.find("Worker:\n      Replicas:        ")
            startnumber = worker_replicas+len("Worker:\n      Replicas:        ")
            replicas = describepod[startnumber:startnumber+1]


            fconf = open("configperjob/" + id + ".txt", "w")
            fconf.write(retconfig)
            fconf.write(resourceslist)
            fconf.write("    WORLD_SIZE:        " + str(int(replicas)+1))
            fconf.close()

            fpod = open("describepodsterminated/" + id + ".txt", "w")
            fpod.write(describepod)
            fpod.close()


fworker = open("epochdata/idorder.txt", "w")
for id in foundids:
    fworker.write(id + "\n")
fworker.close()

print(nrran)