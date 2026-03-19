import os
import random
import time
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
import absl.testing.absltest
import tests.Examples.openfhe.ckks.rotom.mnist.mnist_rotom_openfhe_pybind as mnist

MODEL_PATH = "tests/Examples/common/mnist/data/traced_model.pt"
DATA_PATH = "tests/Examples/common/mnist/data"


def read_from_directory(dirpath: str) -> Dict[str, List[float]]:
  """Reads all .npz files from a directory and maps filename to list of floats.

  File format: compressed numpy .npz files with a 'data' key containing the
  array.
  Returns a dict from filename (with extension) to list of floats.
  """
  result = {}

  if not os.path.isdir(dirpath):
    print(f"Error: Could not open directory {dirpath}")
    return result

  for filename in os.listdir(dirpath):
    if filename == "." or filename == "..":
      continue

    if not filename.endswith(".npz"):
      continue

    fullpath = os.path.join(dirpath, filename)
    try:
      npz_data = np.load(fullpath)
      if "data" not in npz_data:
        print(f"Warning: 'data' key not found in {fullpath}")
        continue

      # Extract data array and flatten to 1D list
      data_array = npz_data["data"]
      values = data_array.flatten().tolist()
      result[filename] = values
      npz_data.close()
    except Exception as e:
      print(f"Warning: Could not load file {fullpath}: {e}")
      continue

  return result


class RotomMNISTTestDataset(Dataset):
  """This custom dataset loads the raw MNIST test data and labels

  from the files specified by `data_root`.
  It applies the Normalize transform manually during loading.
  """

  def __init__(
      self,
      data_root: str,
  ):
    self.data_root = data_root

    labels_path = os.path.join(self.data_root, "t10k-labels-idx1-ubyte")
    with open(labels_path, "rb") as f:
      # Skip header
      f.read(8)
      labels = np.frombuffer(f.read(), dtype=np.uint8)
      self.targets = torch.tensor(labels, dtype=torch.long)

    # Read Rotom packed inputs from inputs/ directory
    inputs_map = read_from_directory(
        "tests/Examples/openfhe/ckks/rotom/mnist/inputs"
    )
    self.images = inputs_map["mlp_mnist_inputs.npz"]
    self.weights = {}
    self.weights["3.npz"] = inputs_map["3.npz"]
    self.weights["21.npz"] = inputs_map["21.npz"]
    self.weights["23.npz"] = inputs_map["23.npz"]
    self.weights["26.npz"] = inputs_map["26.npz"]

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = self.images[index]
    label = self.targets[index]
    return image, label


class MNISTTest(absl.testing.absltest.TestCase):

  def test_run_test(self):
    test_dataset = RotomMNISTTestDataset(data_root=DATA_PATH)

    crypto_context = mnist.mnist__generate_crypto_context()
    key_pair = crypto_context.KeyGen()
    public_key = key_pair.publicKey
    secret_key = key_pair.secretKey
    crypto_context = mnist.mnist__configure_crypto_context(
        crypto_context, secret_key
    )

    # 4. Evaluation Loop
    total = 4
    correct = 0

    # choose 4 random images from the test dataset
    random_samples = random.sample(test_dataset, 4)

    for image, label in random_samples:
      input_encrypted = mnist.mnist__encrypt__arg0(
          crypto_context, image, public_key
      )

      start_time = time.time()
      output_encrypted = mnist.mnist(
          crypto_context,
          input_encrypted,
          test_dataset.weights["3.npz"],
          test_dataset.weights["21.npz"],
          test_dataset.weights["23.npz"],
          test_dataset.weights["26.npz"],
      )
      end_time = time.time()

      time_elapsed_ms = (end_time - start_time) * 1000.0
      print(f"CPU time used: {time_elapsed_ms:.2f} ms")

      output = mnist.mnist__decrypt__result0(
          crypto_context, output_encrypted, secret_key
      )
      non_zero_results = [result for result in output if result != 0]
      guessed_label = non_zero_results.index(max(non_zero_results))

      if guessed_label == label.item():
        correct += 1

      print(f"guessed_label: {guessed_label}, label: {label.item()}")

    self.assertGreaterEqual(correct, 0.75 * total)
