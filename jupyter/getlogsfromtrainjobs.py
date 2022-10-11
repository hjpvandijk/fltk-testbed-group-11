import os

# ret = os.popen("kubectl get configmap -n test --sort-by=.metadata.creationTimestamp --no-headers -o custom-columns=:.metadata.name").read()

ret = os.popen("kubectl get pods -n test --sort-by=.metadata.creationTimestamp --no-headers -o custom-columns=:.metadata.name").read()


jobs = ret[ret.find("fl-server")+len("fl-server")+1:]

splitted = jobs.split('\n')

isExist = os.path.exists("epochdata")
if not isExist:
    os.mkdir("epochdata")

for i in splitted:
    if "master" in i:
        
        retlogs = os.popen("kubectl logs -n test " + i).read()
        epochdata = retlogs[retlogs.find("[EpochData"):]
        f = open("epochdata/" + i + "_epochdata.txt", "w")
        f.write(epochdata)
        f.close()


