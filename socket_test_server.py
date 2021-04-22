from socket_com import ServerTCP, ServerUDP

server = ServerTCP(SERVER="10.217.22.85", MSG_SIZE=100000, DELAY=5e-1)
# server = ServerUDP(SERVER="10.217.22.85", MSG_SIZE=25e4,  CHUNK=1000, DELAY=1000e-6)

server.start()
