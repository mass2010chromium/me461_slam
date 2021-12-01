import serial
import struct
import numpy as np
import sys
fname = sys.argv[1]

ser = serial.Serial('COM4', 230400, timeout=10)

def update_func():
    N = 5000
    data = np.zeros((N, 4))
    print("alloc")
    garbage = ser.readline()
    print("sync")
    for i in range(N):
        read_bytes = ser.read(16)
        # [u, left, right] = struct.unpack('ff', read_bytes)
        data[i, :] = struct.unpack('ffff', read_bytes)
        index = data[i, -1]
        if i != int(index):
            print("Data desync", i, data[i, :])
            return
        #print(i)

    np.savetxt(fname, data, delimiter=",")

update_func()
