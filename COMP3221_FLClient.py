import sys
import socket
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

class Client:
    def __init__(self, client_id, port, opt_method):
        self.client_id = client_id
        self.port = port
        self.opt_method = opt_method

        self.X_train, self.y_train, self.X_test, self.y_test = self.load_dataset()
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        self.batch_size = 64
        if self.opt_method == 0:
            self.batch_size = len(self.train_data)
        self.trainloader = DataLoader(self.train_data, batch_size = self.batch_size)
        self.testloader = DataLoader(self.test_data, batch_size = len(self.test_data))

        self.model = self.LinearRegressionModel()
        self.loss = nn.MSELoss()
        self.learning_rate = 0.00025
        self.epochs = 2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size = 8):
            super(self.__class__, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            output = self.linear(x)
            return output.reshape(-1)

    def update_model(self, global_model_state):
        self.model.load_state_dict(global_model_state)

    def train(self):
        self.model.train()
        total_loss = 0
        total_batches = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0
            for _, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            epoch_loss = epoch_loss / batch_count
            total_loss += epoch_loss
            total_batches += 1
        average_loss = total_loss / total_batches
        return average_loss

    def log_result(self, mse_train, mse_test, round):
        mse_train = self.train()
        mse_test = self.test()
        log_file = f"{self.client_id}_log.txt"
        with open(log_file, 'a') as f:
            f.write(f"Global Iteration: {str(round)}\n")
            f.write(f"Training MSE: {str(mse_train)}\n")
            f.write(f"Testing MSE: {str(mse_test)}\n\n")
    
    def test(self):
        self.model.eval()
        mse = 0
        total_batches = 0
        for x, y in self.testloader:
            y_pred = self.model(x)
            mse += self.loss(y_pred, y).item()
            total_batches += 1
        if total_batches > 0:
            mse = mse / total_batches
        return mse

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

        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.float32)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.float32)

        return X_train, y_train, X_test, y_test

    def start(self):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.bind(('localhost', self.port))
        client.connect(('localhost', 6000))
        print("connected to server")
        packet = {'client_id': self.client_id, 'sample_size': len(self.train_data)}
        encoded_packet = pickle.dumps(packet)
        client.send(encoded_packet)
        count = 0
        while True:
            print(f"I am {self.client_id}")
            global_model_state = pickle.loads(client.recv(4096))
            if global_model_state == "finished":
                break
            print("Received new global model")
            self.update_model(global_model_state)
            mse = self.test()
            print(f"Testing MSE: {str(mse)}")
            print("Local training ...")
            loss = self.train()
            self.log_result(loss, mse, count+1)
            print(f"Training MSE: {str(loss)}")
            print("Sending new local model")
            client.send(pickle.dumps(self.model.state_dict()))
            count+=1
        client.close()

if __name__ == "__main__":
    # python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(1)

    client_id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = int(sys.argv[3])
    client = Client(client_id, port, opt_method)
    filename = f"{client_id}_log.txt"
    with open(filename, 'w') as f:
        f.truncate(0)
    client.start()
