import os

from numpy import arange

for i in arange(0.49145, 0.4918, 0.0001):
    print("==> test 998th epoch_wp{}".format(i))
    wp = "wp_"+str(i)
    name = "transt998_"+wp
    os.system("python -u pysot_toolkit/autotune.py --name {0} --window_penalty {1}".format(name, i))

