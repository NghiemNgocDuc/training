import sys
import numpy as np
import h5py
from sys import stdout

aqm = h5py.File(sys.argv[1], 'r') # reading the HDF5 file

o1 = open(sys.argv[2], 'w') # saving property information

AQMmol_ids = list(aqm.keys())

for molid in AQMmol_ids:

  stdout.write('Current molecule: '+molid+'\n  Conformations:')

  AQMconf_ids = list(aqm[molid].keys())

  for confid in AQMconf_ids:

    Z = np.array(aqm[molid][confid]['atNUM'])
    EAT = float(list(aqm[molid][confid]['eAT'])[0])
    EMBD = float(list(aqm[molid][confid]['eMBD'])[0])
    EGAP = float(list(aqm[molid][confid]['HLgap'])[0])
    EH = float(list(aqm[molid][confid]['eH'])[0])
    EL = float(list(aqm[molid][confid]['eL'])[0])
    DIP = float(list(aqm[molid][confid]['DIP'])[0])
    POL = float(list(aqm[molid][confid]['mPOL'])[0])

    o1.write('{:26s}'.format(confid) + '{:12f}'.format(int(len(Z))) + '{:16f}'.format(EAT) + "{:16f}".format(EMBD) + "{:16f}".format(DIP) + "{:16f}".format(POL) + "{:16f}".format(EGAP) + "{:16f}".format(EH) + "{:16f}".format(EL)  + "\n")

o1.close()
