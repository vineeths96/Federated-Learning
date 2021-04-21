from socket_com import ServerTCP, ServerUDP

server = ServerTCP(SERVER="10.217.22.85", MSG_SIZE=100000)
# server = ServerUDP(SERVER="10.217.22.85", MSG_SIZE=25e4, DELAY=1e-6)

server.start()
