from socket_com.TCPSocket import TCPServer
from socket_com.UDPSocket import UDPServer


config = dict(
    num_epochs=1,
    batch_size=128,
    # communication = "TCP",
    communication="UDP",
    server_address = "10.32.50.26",
    # communication="UDP",
    # architecture="ResNet50",
    architecture="VGG16",
    message_size = {"ResNet50": 23520842, "VGG16": 14728266},
    local_steps=1,
    chunk=2000,
    delay=10e-6,
    K=10000,
    # compression=1/1000,
    # quantization_level=6,
    # higher_quantization_level=10,
    # quantization_levels=[6, 10, 16],
    # rank=1,
    reducer="GlobalRandKReducer",
    seed=42,
    log_verbosity=2,
    lr=0.1,
)

if config['communication'] == "TCP":
    server = TCPServer(SERVER=config['server_address'], MSG_SIZE=config['message_size'][config['architecture']], DELAY=config['delay'])
elif config['communication'] == "UDP":
    server = UDPServer(SERVER=config['server_address'], MSG_SIZE=config['message_size'], CHUNK=config['chunk'], DELAY=config['delay'])
else:
    raise NotImplementedError("Communication method not implemented.")

server.start()
