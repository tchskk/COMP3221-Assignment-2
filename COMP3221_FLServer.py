import sys
import socket
import time
import json
import select

def load_config(filename):
    # neighbours = {} of "(node_id, port)" -> (Node object, cost)
    # (new) neigbours = {'neighbour': {'cost': cost, 'port': port}}
    neighbours = {}
    f = open(filename, 'r')
    f.readline()
    lines = f.read().split('\n')
    for line in lines:
        word = line.split(" ")
        node = word[0]
        cost = float(word[1])
        port = int(word[2])
        neighbours[node] = {'cost': cost, 'port': port, 'heartbeat': 11}
    return neighbours

def main(node, port, neighbours):
    node = Node(node, port, neighbours)
    node.start()
    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(1)

    node = sys.argv[1]
    port = int(sys.argv[2])
    filename = sys.argv[3]
    neighbours = load_config(filename)
    main(node, port, neighbours)
