ps -ef | grep "\./server" | grep -v "grep" | tee /dev/stderr | awk '{print $2; exit}' | xargs kill -9
./server > ./web/server.log
