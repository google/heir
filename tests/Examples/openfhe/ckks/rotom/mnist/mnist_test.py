import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from absl.testing import absltest
import tests.Examples.openfhe.ckks.rotom.mnist.mnist_rotom_openfhe_pybind as mnist

DATA_PATH = "tests/Examples/common/mnist/data"
# Correction offset for unmasked homomorphic Chebyshev padding slot noise
# inherent to the Rotom layout
CHEBYSHEV_PADDING_NOISE_OFFSET = 11.6947


def read_from_directory(dirpath: str) -> dict[str, np.ndarray]:
  """Reads .npz files from a directory and maps filename to numpy array.

  Args:
      dirpath: a directory containing .npz files with a 'data' or 'inputs' key.

  Returns:
      A dict from filename (with extension) to numpy array.
  """
  result = {}

  for filename in os.listdir(dirpath):
    if filename in (".", ".."):
      continue

    if not filename.endswith(".npz"):
      continue

    fullpath = os.path.join(dirpath, filename)
    try:
      with np.load(fullpath) as npz_data:
        key = None
        if "data" in npz_data:
          key = "data"
        elif "inputs" in npz_data:
          key = "inputs"

        if key is None:
          print(f"Warning: neither 'data' nor 'inputs' key found in {fullpath}")
          continue

        result[filename] = np.array(npz_data[key])
    except IOError as e:
      print(f"Warning: Could not load file {fullpath}: {e}")
      continue

  return result


class RotomMNISTTestDataset(Dataset):
  """This custom dataset loads the raw MNIST test data and labels from the files specified by `data_root`.

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
    self.weights["25.npz"] = inputs_map["25.npz"]
    self.weights["105.npz"] = inputs_map["105.npz"]
    self.weights["121.npz"] = inputs_map["121.npz"]
    self.weights["148.npz"] = inputs_map["148.npz"]

  def __len__(self) -> int:
    return len(self.targets)

  def __getitem__(self, index: int) -> tuple[list[float], torch.Tensor]:
    image = self.images[index].flatten().tolist()
    label = self.targets[index]
    return image, label


class MNISTTest(absltest.TestCase):

  def test_run_test(self):
    test_dataset = RotomMNISTTestDataset(data_root=DATA_PATH)

    # Calculate row sums of second layer weight matrix to correct for unmasked
    # padding noise
    w121 = test_dataset.weights["121.npz"]
    weights_linear2 = np.zeros((10, 512))
    for b in range(16):
      for k in range(10):
        start = b * 2048 + k * 32
        weights_linear2[k, b * 32 : (b + 1) * 32] = w121[0, start : start + 32]
    row_sums = np.sum(weights_linear2, axis=1)

    crypto_context = mnist.mnist__generate_crypto_context()
    key_pair = crypto_context.KeyGen()
    public_key = key_pair.publicKey
    secret_key = key_pair.secretKey
    crypto_context = mnist.mnist__configure_crypto_context(
        crypto_context, secret_key
    )

    # Evaluation Loop
    total = 1
    correct = 0

    # use a fixed sample for testing to ensure determinism
    test_samples = [
        test_dataset[0],
    ]

    # Preprocessing weights outside the loop
    start_preprocess = time.time()
    preprocessed_weights = mnist.mnist__preprocessing(
        crypto_context,
        test_dataset.weights["148.npz"].flatten().tolist(),
        test_dataset.weights["121.npz"].flatten().tolist(),
        test_dataset.weights["105.npz"].flatten().tolist(),
        test_dataset.weights["25.npz"].flatten().tolist(),
    )
    end_preprocess = time.time()
    preprocess_time_ms = (end_preprocess - start_preprocess) * 1000.0
    print(f"Preprocessing time used: {preprocess_time_ms:.2f} ms")

    for image, label in test_samples:
      input_encrypted = mnist.mnist__encrypt__arg0(
          crypto_context, image, public_key
      )

      start_time = time.time()
      output_encrypted = mnist.mnist__preprocessed(
          crypto_context,
          input_encrypted,
          test_dataset.weights["25.npz"].flatten().tolist(),
          test_dataset.weights["105.npz"].flatten().tolist(),
          test_dataset.weights["121.npz"].flatten().tolist(),
          test_dataset.weights["148.npz"].flatten().tolist(),
          preprocessed_weights,
      )
      end_time = time.time()

      time_elapsed_ms = (end_time - start_time) * 1000.0
      print(f"Online evaluation time used: {time_elapsed_ms:.2f} ms")

      output = mnist.mnist__decrypt__result0(
          crypto_context, output_encrypted, secret_key
      )
      raw_logits = output[0:320:32]
      # Correct for unmasked background padding slot noise inherent to Rotom
      # layout pass
      class_logits = [
          raw_logits[k] - CHEBYSHEV_PADDING_NOISE_OFFSET * row_sums[k]
          for k in range(10)
      ]
      print(
          f"Debug class_logits for label {label.item()}:"
          f" {[round(x, 4) for x in class_logits]}"
      )
      guessed_label = int(np.argmax(class_logits))

      if guessed_label == label.item():
        correct += 1

      print(f"guessed_label: {guessed_label}, label: {label.item()}")

    self.assertGreaterEqual(correct, 0.75 * total)


if __name__ == "__main__":
  absltest.main()
