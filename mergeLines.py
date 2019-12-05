import matplotlib.pyplot as plt

import numpy as np

def merge_lines(rholines, num):
    lines = []
    for i in range(num):
        lines.append((rholines[i,0,0], rholines[i,0,1]))
    
    
    dtype = [('rho', float), ('theta', float)]
    lines = np.array(lines, dtype=dtype)
    #print(lines)
    lines = np.sort(lines, order=['theta'])
    print(lines)
    print(np.average(lines, axis=0))
    print(lines.dtype)
    #plt.scatter(lines[:,0], lines[:,1], alpha=0.5)
    #plt.show()
    return merged_lines