import bluetooth
import urllib.request

hostMACAddress = "dc:a6:32:ef:78:96" # The address of Raspberry PI Bluetooth adapter on the server. The server might have multiple Bluetooth adapters.

port = 1
backlog = 1
size = 1024
s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
s.bind((hostMACAddress, port))
s.listen(backlog)
print("listening on port ", port)

import struct

def handle_request():
    try:
        client, clientInfo = s.accept()
        print("server recv from: ", clientInfo)
        data = client.recv(size)
        message_type = chr(data[0])
        body_size = data[1]
        message_body = data[2:2+body_size].decode('utf-8')
        print(message_type, body_size, message_body)
        if message_type == 'g':
            resp = urllib.request.urlopen(f"http://localhost:8080/{message_body}").read()
        elif message_type == 'p':
            json_body = data[2+body_size:]
            print(json_body)
            req = urllib.request.Request(f"http://localhost:8080/{message_body}", data=json_body)
            resp = urllib.request.urlopen(req).read()
        else:
            client.send(b'error')
            return
        if resp:
            print(resp)
            client.send(resp) # Echo back to client
    finally:
        print("Closing socket")
        client.close()

while True:
    handle_request()
