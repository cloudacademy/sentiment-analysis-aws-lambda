""" Manually load OS libraries """
import os
import ctypes

for d, dirs, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))
