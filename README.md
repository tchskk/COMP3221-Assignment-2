# COMP3221-Assignment-2
SID : 520060531

This `README.md` file contains the instructions on how to execute and run our Client and Server Python program.

## Files

`COMP3221_FLClient.py` : Client Program

    python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
\<Client-id> : `client1` to `client5`

\<Port-Client> : `6001` to `6005`

\<Opt-Method> : `0` for Gradient Descent and `1` for Mini-batch Gradient Descent

`COMP3221_FLServer.py`: Server Program

    python COMP3221_FLClient.py <Port-Server> <Sub-Client>
\<Port-Server> : `6000`

\<Sub-Client> : `0` for all clients and `1` to `5` for subsampling, where the server randomly aggregates models from only M out of the K clients

`client{}_log.txt`:  Contains all the Training and Testing MSEs for the client from previous run, will reset for every run.


## Instructions
Firstly, store all the csv files in the folder `FLData` or the program will not work

    ./FLData
        callhousing_test_client1.csv
        callhousing_train_client1.csv
        callhousing_test_client2.csv
        callhousing_train_client2.csv
        callhousing_test_client3.csv
        callhousing_train_client3.csv
        callhousing_test_client4.csv
        callhousing_train_client4.csv
        callhousing_test_client5.csv
        callhousing_train_client5.csv

### Libraries used

Make sure to have the libraries/packages below installed :

    import sys
    import socket
    import pickle
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np
    import threading
    import time
    import random

### Execution of Program

1. Using the file syntax from above, execute each client and server in its own terminal
2. Wait 30 seconds after first client connects then global iteration will start
3. Wait until total global iterations reached then program will stop for each terminal
4. For maunal evaluation, check the terminal output or the log files

### Thank you and have fun :)
