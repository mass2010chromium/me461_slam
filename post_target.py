from urllib import request
import sys
import json

x = float(sys.argv[1])
y = float(sys.argv[2])
t = float(sys.argv[3])

data = json.dumps({'x': x, 'y': y, 't': t}).encode('utf-8')
req = request.Request("http://localhost:8080/target", data=data)
resp = request.urlopen(req)
