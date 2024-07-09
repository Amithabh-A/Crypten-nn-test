import crypten
import torch
from crypten import mpc

ALICE = 0
BOB = 1

count = 100

@mpc.run_multiprocess(world_size=2)
def encrypt_model_and_data():
    # Load pre-trained model to Alice
    plaintext_model = crypten.load_from_party('tutorial4_alice_model.pth', src=ALICE)

    # Encrypt model from Alice
    dummy_input = torch.empty((1, 784))
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt(src=ALICE)

    # Load data to Bob
    data_enc = crypten.load_from_party('/tmp/bob_test.pth', src=BOB)
    data_enc2 = data_enc[:count]
    data_flatten = data_enc2.flatten(start_dim=1)

    # Classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_flatten)

    # Verify the results are encrypted:
    crypten.print("Output tensor encrypted:", crypten.is_encrypted_tensor(output_enc))

    # Decrypting the result
    output = output_enc.get_plain_text()

    # Obtaining the labels
    pred = output.argmax(dim=1)
    crypten.print("Decrypted labels:\n", pred)

encrypt_model_and_data()
