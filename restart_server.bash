ps -ef | grep "\./server" | tee /dev/stderr | awk '{print $2; exit}' | xargs kill -9
./server
