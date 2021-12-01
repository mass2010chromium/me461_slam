import serial

ser = serial.Serial('COM4', 115200, timeout=1)

def update_func():
    while True:
        in_line = ser.readline()
        print(in_line.decode('utf-8'), end="")

update_func()
