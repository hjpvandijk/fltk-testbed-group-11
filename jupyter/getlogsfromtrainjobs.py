import os

# ret = os.popen("kubectl get configmap -n test --sort-by=.metadata.creationTimestamp --no-headers -o custom-columns=:.metadata.name").read()

ret = os.popen("kubectl get pods -n test --sort-by=.metadata.creationTimestamp --no-headers -o custom-columns=:.metadata.name").read()


jobs = ret[ret.find("fl-server")+len("fl-server")+1:]

splitted = jobs.split('\n')

isExist = os.path.exists("epochdata")
if not isExist:
    os.mkdir("epochdata")

isExist = os.path.exists("describepods")
if not isExist:
    os.mkdir("describepods")

foundids = []
workers = []
nrran = 0

for i in splitted:
    if "worker" in i:
        id = i[len("trainjob-"):len(i)-len("-worker-0")]
        if id not in foundids:
            foundids.append(id)
            workers.append(i)
            print(i)
            nrran += 1
            retlogs = os.popen("kubectl logs -n test " + i).read()
            epochdata = retlogs[retlogs.find("[EpochData"):]
            f = open("epochdata/" + i + "_epochdata.txt", "w")
            f.write(epochdata)
            f.close()

            retconfig = os.popen("kubectl describe configmap -n test worker-" + id + "-0").read()
            retsysconfig = os.popen("kubectl get pod " + i + " -o yaml -n test").read()
            resources = retsysconfig.find("resources")
            terminationmessagepath = retsysconfig.find("terminationMessagePath")
            resourceslist = retsysconfig[resources:terminationmessagepath]
            fconf = open("configperjob/" + id + ".txt", "w")
            fconf.write(retconfig)
            fconf.write(resourceslist)
            fconf.close()

            describepod = os.popen("kubectl describe pod " + i + " -n test").read()
            fpod = open("describepods/" + id + ".txt", "w")
            fpod.write(describepod)
            fpod.close()


fworker = open("epochdata/idorder.txt", "w")
for id in foundids:
    fworker.write(id + "\n")
fworker.close()

print(nrran)