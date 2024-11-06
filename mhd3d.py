#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:41:10 2024

@author: stephi
"""

import numpy as np
import matplotlib.pyplot as plt

# Example of loading or simulating S-index data
time = np.linspace(0, 22, 500)  # 22-year cycle, similar to solar cycle
s_index = 0.16 + 0.02 * np.sin(2 * np.pi * time / 11)  # Simulated S-index
def s_index_to_log_rhk(s):
    return -4.67 + 0.78 * np.log10(s)

log_rhk = s_index_to_log_rhk(s_index)
x, y, z = np.meshgrid(np.linspace(-1, 1, 20), 
                      np.linspace(-1, 1, 20), 
                      np.linspace(-1, 1, 5))
def magnetic_field_with_activity(x, y, z, s):
    Bx = -y * s
    By = x * s
    Bz = np.zeros_like(x)
    return Bx, By, Bz
from mpl_toolkits.mplot3d import Axes3D

# Assume time[100] corresponds to a specific point in the cycle
current_s_index = s_index[100]

# Compute magnetic field with activity modulation
Bx, By, Bz = magnetic_field_with_activity(x, y, z, current_s_index)

# Plot as before
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x, y, z, Bx, By, Bz, length=0.1, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Magnetic Field with Solar Activity Modulation')
plt.show()
