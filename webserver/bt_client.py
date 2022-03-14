import bluetooth

host = "DC:A6:32:EF:78:96" # The address of Raspberry PI Bluetooth adapter on the server.
port = 1
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((host, port))

import sys

header = 'g'
path = "pose_slam"
data = ""
if len(sys.argv) > 1:
    path = sys.argv[1]
if len(sys.argv) > 2:
    data = sys.argv[2]
    header = 'p'

if len(path) > 255:
    print("Path length exceeds 255 bytes")
    sys.exit(1)

text = header + chr(len(path)) + path + data

sock.send(text.encode('utf-8'))

data = sock.recv(1024)
while not data:
    data = sock.recv(1024)
print("from server: ", data)

sock.close()


