fid = open("epochdata/idorder.txt")
ids = fid.read().split("\n")

for id in ids:
    if id != "":
        fpod = open("describepods/" + id + ".txt", "r")
        podread = fpod.read()
        world_size = podread[podread.find("WORLD_SIZE"):podread.find("RANK")]
        fconf = open("configperjob/" + id + ".txt", "a")
        fconf.write(world_size)
        fconf.close()