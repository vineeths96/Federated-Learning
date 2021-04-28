from socket_com import ServerTCP, ServerUDP

# server = ServerTCP(SERVER="10.216.18.179", MSG_SIZE=100000, DELAY=0)
# server = ServerUDP(SERVER="10.216.18.179", MSG_SIZE=100000, CHUNK=1000, DELAY=1e-6)
server = ServerUDP(SERVER="10.32.50.26", MSG_SIZE=100000, CHUNK=10 * 2 * 250, DELAY=0e-6)

server.start()
