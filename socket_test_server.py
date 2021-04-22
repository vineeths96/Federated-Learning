from socket_com import ServerTCP, ServerUDP

# server = ServerTCP(SERVER="10.216.18.179", MSG_SIZE=100000, DELAY=0)
server = ServerUDP(SERVER="10.216.18.179", MSG_SIZE=100000, CHUNK=1000, DELAY=1e-6)

server.start()
