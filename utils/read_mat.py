import mmap
import numpy as np

"""
Convenience class for reading from shared memory.
"""
class SharedArray:

    """
    Set up a shared memory reader.
    @param filename : shared memory filename.
    @param dimensions : Tuple of dimensions of the array.
    @param dtype : Data type of the array.
    """
    def __init__(self, filename, dimensions, dtype):
        self.filename = filename
        self.dimensions = dimensions
        self.f = open(filename, "r+b")
        self.shm = mmap.mmap(self.f.fileno(), 0)
        self.dtype = dtype
        self.size = np.prod(dimensions) * np.dtype(dtype).itemsize

    def close(self):
        self.f.close()
        self.shm.close()

    def read(self):
        self.shm.seek(0)
        buffer = self.shm.read(self.size)
        return True, np.frombuffer(buffer, dtype=self.dtype, count=self.size, offset=0).reshape(self.dimensions)

if __name__ == "__main__":
    import cv2
    videocap = SharedArray("../.webserver.video", [480, 640, 3], dtype=np.uint8)
    while True:
        success, data = videocap.read()
        cv2.imshow("video", data)
        cv2.waitKey(1)
