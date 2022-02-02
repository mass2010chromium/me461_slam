import serial
import struct
import time

ser = serial.Serial('/dev/ttyAMA1', 115200, timeout=10)
garbage = ser.readline()
print("sync", garbage)
i = 0
cmd_vel = 0
initialized = False
init_count = 10
save_start = 0
x0 = 0
y0 = 0
while 1:
    read_bytes = ser.read(25)
    #print("read")
    data = struct.unpack('fffffLc', read_bytes)
    index = data[-2]
    if data[-1] != b'\n':
        if i != -1:
            i = -1
            print("Data desync", i, data)
        garbage = ser.readline()
        continue
    if i == -1:
        print("sync", index)
        i = index
        continue
    #elif i != index:
    #    continue

    x, y, heading = data[:3]
    if not initialized:
        x0 += x
        y0 += y
        save_start += 1
        if save_start == init_count:
            x0 /= init_count
            y0 /= init_count
            initialized = True
            print("Initialized")
        continue
    if abs(x - x0) > 10 or abs(y - y0) > 10:
        continue
    if (x-x0)**2 + (y-y0)**2 > 1:
        print("Done", x, y, x0, y0)
        break
    command = struct.pack('ffc', 1.0, 0.0, '\n'.encode('utf-8'))
    ser.write(command)
