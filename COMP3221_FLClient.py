import sys
import socket
import json
import select

class Client:
    def __init__(self, client_id, port, opt_method):
        self.client_id = client_id
        self.port = port
        self.opt_method = opt_method
        self.training_dataset = None
        self.testing_dataset = None

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
        # packet = {'node': 'A', 'neighbours': {'B':(via, cost B to A)}}
        neighbour = decoded_packet['node']
        neighbour_table = decoded_packet['neighbours']
        self.neighbours[neighbour]['heartbeat'] = 11
        if decoded_packet.get('link_cost_update') != None:
            if float(decoded_packet['link_cost_update']) != self.neighbours[neighbour]['cost']:
                self.neighbours[neighbour]['cost'] = float(decoded_packet['link_cost_update'])
                self.update_reachability_matrix(neighbour, neighbour_table)
                self.update = True
        else:
            self.update_reachability_matrix(neighbour, neighbour_table)

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
        training_dataset = []
        testing_dataset = []

        f = open(f"./FLData/calhousing_train_{self.client_id}.csv", 'r')
        f.readline()
        lines = f.read().split('\n')
        training_dataset = [0]*len(lines)
        f.close()
        for i in range(len(lines)):
            splitted = lines[i].split(",")
            if len(splitted) == 9:
                training_dataset[i] = list(map(float, splitted))

        f = open(f"./FLData/calhousing_test_{self.client_id}.csv", 'r')
        f.readline()
        lines = f.read().split('\n')
        testing_dataset = [0]*len(lines)
        f.close()
        for i in range(len(lines)):
            splitted = lines[i].split(",")
            if len(splitted) == 9:
                testing_dataset[i] = list(map(float, splitted))

        return training_dataset, testing_dataset

    def start(self):
        self.training_dataset, self.testing_dataset = self.load_dataset()
        print(self.training_dataset[0:5])
        print(self.testing_dataset[0:5])


if __name__ == "__main__":
    # python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(1)

    client_id = sys.argv[1]
    port = int(sys.argv[2])
    opt_method = sys.argv[3]
    client = Client(client_id, port, opt_method)
    client.start()
