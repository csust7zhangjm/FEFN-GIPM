import os
ep_id = [826, 832, 836, 842, 844, 846, 862, 864, 866, 868, 872, 878, 880, 886, 892, 898, 900, 902, 906, 908, 914, 916]

"""for i in range(850, 1002, 2):
    print("==> test {}th epoch".format(i))
    name = "transt"+str(i)
    os.system("python -u pysot_toolkit/test_all_ep_1.py --name {0} --epnum {1}".format(name, i))"""


for i in range(len(ep_id)):
    print("==> test {}th epoch".format(ep_id[i]))
    name = "transt"+str(ep_id[i])
    os.system("python -u pysot_toolkit/test_all_ep_1.py --name {0} --epnum {1}".format(name, ep_id[i]))
