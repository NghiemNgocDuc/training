import os
from pandas import DataFrame

folder_path = "../Data/THR_data_19SB_OPC_GBn2"
from util import *


if __name__ == "__main__":
    Dataset = []
    all_u = []
    feats = []
    feat_names = []
    sample_dirs = os.listdir(os.path.join(folder_path, "WINDOWS"))
    filecount = 0
    for line in open(os.path.join(folder_path, "embeddings.out")):
        toks = list(filter(None, line.strip().split(" ")))
        if len(toks) < 10:
            continue
        if toks[0] == "ATOM":
            feat_names = toks
        else:
            feats.append(toks)
    print(feat_names)
    print(feats)
    attr = DataFrame(feats,columns=feat_names)
    print(attr)

    path = os.path.join(folder_path, "WINDOWS", sample_dirs[0],"coordinates.xyz")
    abc =readXYZ(path)
    print(abc)

