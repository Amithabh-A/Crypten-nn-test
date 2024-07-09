import crypten
import torch
from utils import compute_accuracy
ALICE = 0
BOB = 1

crypten.init()
torch.set_num_threads(1)

from alicenet import AliceNet
crypten.common.serial.register_safe_class(AliceNet)

ALICE = 0
BOB = 1

# Encrypting a pretrained model

# Load pre-trained model to Alice
dummy_model = AliceNet()
plaintext_model = torch.load('tutorial4_alice_model.pth')
# print(plaintext_model)

# Encrypt the model from Alice:

# 1. Create a dummy input with the same shape as the model input
dummy_input = torch.empty((1, 784))
# 2. Construct a CrypTen network with the trained model and dummy_input
private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
# 3. Encrypt the CrypTen network with src=ALICE
private_model.encrypt(src=ALICE)

#Check that model is encrypted:
print("Model successfully encrypted:", private_model.encrypted)


# Classifying Encrypted Data with Encrypted Model
import crypten.mpc as mpc
import crypten.communicator as comm
from classification import encrypt_model_and_data as classifier

labels = torch.load('/tmp/bob_test_labels.pth').long()
count = 100 # For illustration purposes, we'll use only 100 samples for classification

classifier()


# Validating Encrypted Classification
from validation import encrypt_model_and_data as validator
validator()
