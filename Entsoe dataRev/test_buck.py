import socket
import os

print("Running on:", socket.gethostname())
print("Available logical CPUs:", os.cpu_count())