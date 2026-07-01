import numpy as np

def readXYZ(file_path):
    XYZ = []
    for line in open(file_path):
        tabs = list(filter(None, line.strip().split(' ')))
        XYZ.append(tabs)
    return np.array(XYZ,dtype=float)

