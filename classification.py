import torch
import crypten
from utils import compute_accuracy

import crypten.mpc as mpc
import crypten.communicator as comm

ALICE = 0
BOB = 1

labels = torch.load('/tmp/bob_test_labels.pth').long()
count = 100 # For illustration purposes, we'll use only 100 samples for classification

@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    # Load pre-trained model to Alice
    model = crypten.load_from_party('tutorial4_alice_model.pth', src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=ALICE)

    # Load data to Bob
    data_enc = crypten.load_from_party('/tmp/bob_test.pth', src=BOB)
    data_enc2 = data_enc[:count]
    data_flatten = data_enc2.flatten(start_dim=1)

    # Classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_flatten)

    # Compute the accuracy
    output = output_enc.get_plain_text()
    accuracy = compute_accuracy(output, labels[:count])
    crypten.print("\tAccuracy: {0:.4f}".format(accuracy.item()))

encrypt_model_and_data()
