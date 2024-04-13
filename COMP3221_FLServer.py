import sys
import socket
import json
import select

import torch
import torch.nn as nn

class Server:
    def __init__(self, port, subsampling):
        self.port = port
        self.subsampling = subsampling
        self.num_glob_iters = 100 # T
        self.global_model = None
        self.clients = {}
        self.no_clients = 0

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size = 8):
            super(self.__class__, self).__init__()
            # Create a linear transformation to the incoming data
            self.linear = nn.Linear(input_size, 1)

        # Define how the model is going to be run, from input to output
        def forward(self, x):
            # Apply linear transformation
            output = self.linear(x)
            return output.reshape(-1)

    def listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('localhost', self.port))
            s.setblocking(0)
            while not self.stop_listen:
                ready = select.select([s], [], [], 1)
                if ready[0]:
                    packet = s.recv(1024)
                    if self.active:
                        self.process_packet(json.loads(packet.decode('utf-8')))

    def process_packet(self, decoded_packet: dict):
        # packet = {  }
        pass

    def send_to(self, s):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            packet = self.generate_packet()
            encoded_packet = json.dumps(packet).encode('utf-8')
            s.sendto(encoded_packet, ('localhost', port))

    def generate_packet(self):
        # packet format
        # packet = {  }
        packet = {}
        return packet
    
    def initialize_connection(self):
        # Connect to Server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 6000))
                s.sendall(b'Hello Boss')
        except:
            print("Can't connect to the Server")
        return s

    def start(self):
        self.global_model = self.LinearRegressionModel()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 6000))
            s.listen(5)

if __name__ == "__main__":
    # python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
    if len(sys.argv) != 3:
        print("Wrong arguments")
        sys.exit(1)

    port = int(sys.argv[1])
    subsampling = sys.argv[3]
    server = Server(port, subsampling)
    server.start()