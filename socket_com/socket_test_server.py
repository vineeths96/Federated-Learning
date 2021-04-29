from TCPSocket import ServerTCP
from UDPSocket import ServerUDP
from parameters import *

# server = ServerTCP(SERVER=SERVER, MSG_SIZE=MSG_SIZE, DELAY=DELAY)
server = ServerUDP(SERVER=SERVER, MSG_SIZE=MSG_SIZE, CHUNK=CHUNK, DELAY=DELAY)

server.start()
