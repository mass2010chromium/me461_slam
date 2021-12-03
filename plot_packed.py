import threading
import sys
import struct

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=10)

buf = deque([])
buf2 = deque([])
buf3 = deque([])
buf4 = deque([])
ts = deque([])
buf_lock = threading.Lock()

def update_func():
    garbage = ser.readline()
    print("sync", garbage)
    i = 0
    while True:
        read_bytes = ser.read(21)
        # [u, left, right] = struct.unpack('ff', read_bytes)
        data = struct.unpack('fffffc', read_bytes)
        index = data[-2]
        if i == -1 and index > 0 and index == int(index):
            print("sync", index)
            i = int(index + 1)
            continue
        if i == -1:
            continue
        elif i != int(index):
            print("Data desync", i, data)
            garbage = ser.readline()
            i = -1
            continue
        with buf_lock:
            if i % 8 == 0 and (len(buf) == 0 or not (abs(data[0] - buf[-1]) > 1
                    or abs(data[1] - buf2[-1]) > 1
                    or abs(data[2] - buf3[-1]) > 1
                    or abs(data[3] - buf4[-1]) > 1)):
                if len(buf) == 1000:
                    buf.popleft()
                    buf2.popleft()
                    buf3.popleft()
                    buf4.popleft()
                    ts.popleft()
                buf.append(data[0])
                buf2.append(data[1])
                buf3.append(data[2])
                buf4.append(data[3])
                ts.append(data[4])
        i += 1

read_thread = threading.Thread(group=None, daemon=True, target=update_func)
read_thread.start()

fig = plt.figure()
ax1, ax2 = fig.subplots(2, 1)
ax1.set_aspect("equal")
ax1.set_xlim([-1, 6])
ax1.set_ylim([-1, 6])
plt.ion()
plt.show()

while True:
    with buf_lock:
        tmp = np.array(buf)
        tmp2 = np.array(buf2)
        tmp3 = np.array(buf3)
        tmp4 = np.array(buf4)
        _ts = np.array(ts)
    #plt.plot(_ts, tmp, label="innovation")
    #plt.plot(_ts, tmp2, label="encoder only")
    if len(tmp2) == 0:
        plt.pause(0.25)
        continue
    ax1.clear()
    ax1.set_title("position")
    ax1.plot(tmp, tmp2, label="pos")
    heading = tmp3[-1]
    xs = (tmp[-1], tmp[-1] + 0.1*np.cos(heading))
    ys = (tmp2[-1], tmp2[-1] + 0.1*np.sin(heading))
    ax1.plot(xs, ys)
    ax1.set_xlim([-1, 6])
    ax1.set_ylim([-1, 6])
    ax1.legend()
    ax2.clear()
    ax2.plot(_ts, tmp3, label="fused")
    ax2.plot(_ts, tmp4, label="encoder")
    ax2.legend()
    ax2.set_title("misc")
    plt.pause(0.25)
