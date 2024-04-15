import sys
import threading
import time
import socket
import pickle

import torch
import torch.nn as nn

import copy

class Server:
    def __init__(self, port, subsampling):
        self.port = port
        self.subsampling = subsampling #0 means all clients, 0-5
        self.num_glob_iters = 1000000 # T
        self.global_model = self.LinearRegressionModel()
        self.clients = {}
        self.client_ids = []
        self.total_datasize = 0

        self.first_handshake = False

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

    def listen(self, server:socket.socket):
        server.listen(5)
        while len(self.client_ids) < 5:
            conn,_ = server.accept()
            print("accepted client")
            if len(self.client_ids) == 0:
                self.first_handshake = True
            packet = conn.recv(1024)
            client_id, sample_size = self.process_packet(pickle.loads(packet))
            self.client_ids.append(client_id)
            self.clients[client_id] = {'connection': conn, 'sample_size': sample_size}
            self.total_datasize += sample_size

    def process_packet(self, decoded_packet: dict):
        # packet = { 'client_id': 'client1', 'batch_size': int() }
        return decoded_packet['client_id'], int(decoded_packet['sample_size'])

    def send_global_model(self, client_connection):
        # packet format
        # packet = {  }
        client_connection.send(pickle.dumps(self.global_model.state_dict()))

    def aggregate_parameters(self, client_model_states: dict):
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])

        for client_id, local_state_dict in client_model_states.items():
            for key in global_state_dict.keys():
                contribution_scale = self.clients[client_id]['sample_size'] / self.total_datasize
                global_state_dict[key] += local_state_dict[key] * contribution_scale

        self.global_model.load_state_dict(global_state_dict)

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', self.port))
        listening_thread = threading.Thread(target = self.listen, args=(server,))
        listening_thread.daemon = True
        listening_thread.start()
        while not self.first_handshake:
            pass
        self.first_handshake = False
        time.sleep(10)

        global_count = 0
        while global_count < self.num_glob_iters:
            count = 0
            client_models = {}
            print("Broadcasting new global model")
            for client_id in self.client_ids:
                conn = self.clients[client_id]['connection']
                conn.send(pickle.dumps(self.global_model.state_dict()))
                #print(self.global_model.state_dict())
                count += 1
            print(f"Global Iteration {global_count}")
            print(f"Total Number of clients: {count}")
            for i in range(count):
                # packet = {'client_id': 'client1, 'local_model': nn.Module, ''}
                client_id = self.client_ids[i]
                print(f"Getting local model from {client_id}")
                conn = self.clients[client_id]['connection']
                local_model = pickle.loads(conn.recv(4096))
                #print(local_model)
                client_models[client_id] = local_model

            self.aggregate_parameters(client_models)
            global_count += 1

        for client_id in self.client_ids:
            message = "finished"
            conn = self.clients[client_id]['connection']
            conn.send(pickle.dumps(message))
            conn.close()

        server.close()

if __name__ == "__main__":
    # python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
    if len(sys.argv) != 3:
        print("Wrong arguments")
        sys.exit(1)

    port = int(sys.argv[1])
    subsampling = int(sys.argv[2])
    server = Server(port, subsampling)
    server.start()