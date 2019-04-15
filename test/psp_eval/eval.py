from subprocess import Popen, PIPE
from threading import Timer
from shutil import copyfile



# dnn.exe write result to psp_exp.csv file
exp_file = "psp_exp.csv"

f0 = open('shit.log', 'w')
f1 = open('fuck.log', 'w')

barriers = ["bsp", "asp", "ssp"]

for barrier in barriers:
    p_server  = Popen(["./dnn.exe", "server", barrier], stdout=f0, stderr=f1)
    p_worker0 = Popen(["./dnn.exe", "w0", barrier], stdout=f0, stderr=f1)
    p_worker1 = Popen(["./dnn.exe", "w1", barrier], stdout=f0, stderr=f1)

    p_list = [p_server, p_worker0, p_worker1]

    try:
        for p in p_list: out, err = p.communicate(timeout=60)
    except Exception as e:
        for p in p_list: p.terminate()

    copyfile(exp_file, ("exp_%s.csv" % barrier))
    open(exp_file, "w").close() #clean the file

f0.close()
f1.close()
