import sys
import socket
import json
import select

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import copy

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

initialize_model = LinearRegressionModel()

class Client:
    def __init__(self, client_id, port, opt_method):
        self.client_id = client_id
        self.port = port
        self.opt_method = opt_method

        self.X_train, self.y_train, self.X_test, self.y_test = self.load_dataset()
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        # Define dataloader for iterable sample over a dataset
        self.batch_size = 25
        if self.opt_method == 0:
            self.batch_size = len(self.train_data)
        self.trainloader = DataLoader(self.train_data, batch_size = self.batch_size)
        self.testloader = DataLoader(self.test_data, batch_size = len(self.test_data))

        self.model = copy.deepcopy(initialize_model)
        self.loss = nn.MSELoss()
        self.learning_rate = 0.01
        self.epochs = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self):
        loss = 0
        self.model.train()
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        mse = 0
        for X, y in self.testing_dataset:
            y_pred = self.model(X)
            # Calculate evaluation metrics
            mse += self.loss(y_pred, y)
            print(str(self.id) + ", MSE of client ",self.id, " is: ", mse)
        return mse

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
    
    def load_dataset(self):
        X_train = None
        y_train = None
        X_test = None
        y_test = None

        f = open(f"./FLData/calhousing_train_{self.client_id}.csv", 'r')
        f.readline()
        lines = f.read().split('\n')
        X_train = []
        y_train = []
        f.close()
        for i in range(len(lines)):
            splitted = lines[i].split(",")
            if len(splitted) == 9:
                X_train.append([float(i) for i in splitted[:8]])
                y_train.append(float(splitted[8]))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
                
        f = open(f"./FLData/calhousing_test_{self.client_id}.csv", 'r')
        f.readline()
        lines = f.read().split('\n')
        X_test = []
        y_test = []
        f.close()
        for i in range(len(lines)):
            splitted = lines[i].split(",")
            if len(splitted) == 9:
                X_test.append([float(i) for i in splitted[:8]])
                y_test.append(float(splitted[8]))
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_train = torch.Tensor(X_train).view(-1,1).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.float32)
        X_test = torch.Tensor(X_test).view(-1,1).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.float32)

        return X_train, y_train, X_test, y_test

    def start(self):
        """with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 6000))
            s.sendall(b'Hello Boss')"""
            

if __name__ == "__main__":
    # python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(1)

    client_id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    client = Client(client_id, port, opt_method)
    client.start()
