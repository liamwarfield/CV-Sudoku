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
    #seperate on theta
    prev = lines[0][1]
    hor = None
    virt = None
    linegrps = []

    for i in range(1, num):
        if abs(lines[i][1] - prev) > .4:
            hor = lines[0:i]
            virt = lines[i:]
            break
        prev = lines[i][1]

    hor = np.sort(hor, order=['rho'])
    virt = np.sort(virt, order=['rho'])
    #print("Horizontal Lines", hor)
    #print("Virtical Lines", virt)
    
    #seperate on rho
    prev = hor[0][0]
    prevgrp = 0
    for i in range(1, len(hor)):
        #print(f"iteration={i} prev={prev} current={hor[i][0]} diff={abs(hor[i][0] - prev)}")
        if abs(hor[i][0] - prev) > 50: # Create a new group
            print(hor[prevgrp:i])
            linegrps.append(hor[prevgrp:i])
            prevgrp = i
        prev = hor[i][0]
    linegrps.append(hor[prevgrp:])
    #print("Horizontal Groups", linegrps)

    prev = virt[0][0]
    prevgrp = 0
    for i in range(1, len(virt)):
        if abs(virt[i][0] - prev) > 50: # Create a new group
            linegrps.append(virt[prevgrp:i])
            prevgrp = i
        prev = virt[i][0]
    linegrps.append(virt[prevgrp:])
    #print("virtizontal Groups", linegrps)
    
    # Take the average of the groups
    merged_lines = []
    for group in linegrps:
        total_rho = 0
        total_theta = 0
        for line in group:
            total_rho += line[0]
            total_theta += line[1]
        avg_rho = total_rho / len(group)
        avg_theta = total_theta / len(group)
        merged_lines.append((avg_rho, avg_theta))
    return merged_lines