
#this code models the 4 layer neural network, to work with normalized fp16 values, generates 10 sets of test inputs in (-1,1) range, a set of weights for each node in (0,1) range, and calculates outputs. All the values are stored in text files with each value in a line. Look for a folder created nn_data to see text files.

import numpy as np
import os

np.random.seed(42)

num_sets = 10
input_size = 10
layer1_size = 8
layer2_size = 6
output_size = 2

inputs = np.random.uniform(-1, 1, size=(num_sets, input_size)).astype(np.float32)

weights1 = np.random.uniform(0, 1, size=(layer1_size, input_size)).astype(np.float32)
weights2 = np.random.uniform(0, 1, size=(layer2_size, layer1_size)).astype(np.float32)
weights3 = np.random.uniform(0, 1, size=(output_size, layer2_size)).astype(np.float32)

def float_to_fp16_uint(arr):
    return arr.astype(np.float16).view(np.uint16)

outputs = []
for inp in inputs:
    layer1 = np.dot(weights1, inp)
    layer2 = np.dot(weights2, layer1)
    out = np.dot(weights3, layer2)
    outputs.append(out.astype(np.float32))
outputs = np.array(outputs)

inputs_fp16 = float_to_fp16_uint(inputs)
weights1_fp16 = float_to_fp16_uint(weights1)
weights2_fp16 = float_to_fp16_uint(weights2)
weights3_fp16 = float_to_fp16_uint(weights3)
outputs_fp16 = float_to_fp16_uint(outputs)

os.makedirs('nn_data', exist_ok=True)

with open('nn_data/inputs.txt', 'w') as f_in:
    for val in inputs_fp16.flatten():
        f_in.write(f"{val:04X}\n")

with open('nn_data/weights.txt', 'w') as f_w:
    for val in np.concatenate([weights1_fp16.flatten(),
                               weights2_fp16.flatten(),
                               weights3_fp16.flatten()]):
        f_w.write(f"{val:04X}\n")

with open('nn_data/outputs.txt', 'w') as f_out:
    for val in outputs_fp16.flatten():
        f_out.write(f"{val:04X}\n")

print("Generated data in 'nn_data' folder with one hex value per line (no '0x' prefix).")


