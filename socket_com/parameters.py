# SERVER="10.216.18.179"
# SERVER_COMP="10.32.50.26"
SERVER="10.32.50.26"
SERVER_COMP="10.216.18.179"

MSG_SIZE=25e5
CHUNK=5000
DELAY=0e-4

BUFFER = 1024 * 16 * 4

use_TCP = True
# use_TCP = False
# UDP_DEBUG = True
UDP_DEBUG = False

REPS = 1
MSG_SIZES = [25e3]
# MSG_SIZES = [1, 5, 10, 50, 100, 500, 1000, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 25e6]