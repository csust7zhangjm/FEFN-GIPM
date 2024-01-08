import os
ep_id = [918, 924, 926, 928, 932, 936, 950, 954, 956, 958, 962, 966, 970, 978, 1000, 981, 955, 971, 917, 995, 957, 899, 889, 823]
"""for i in range(700, 850, 2):
    print("==> test {}th epoch".format(i))
    name = "transt"+str(i)
    os.system("python -u pysot_toolkit/test_all_ep.py --name {0} --epnum {1}".format(name, i))"""
for i in range(len(ep_id)):
    print("==> test {}th epoch".format(ep_id[i]))
    name = "transt"+str(ep_id[i])
    os.system("python -u pysot_toolkit/test_all_ep.py --name {0} --epnum {1}".format(name, ep_id[i]))

