import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from absl.testing import absltest
from tests.Examples.openfhe.ckks.lenet import lenet_interpreter as lenet

# fmt: off
DATA_PATH = "tests/Examples/openfhe/ckks/mnist/data"
MLIR_SRC_PATH = "tests/Examples/openfhe/ckks/lenet/lenet.openfhe.mlir"
# fmt: on


class CustomMNISTTestDataset(Dataset):
  """This custom dataset loads the raw MNIST test data and labels

  from the files specified by `data_root`.
  It applies the Normalize transform manually during loading.
  """

  def __init__(
      self,
      data_root: str,
      *,
      normalize_mean: float = 0.1307,
      normalize_std: float = 0.3081,
  ):
    self.data_root = data_root
    self.normalize_mean = normalize_mean
    self.normalize_std = normalize_std

    labels_path = os.path.join(self.data_root, "t10k-labels-idx1-ubyte")
    try:
      with open(labels_path, "rb") as f:
        # Skip header
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        self.targets = torch.tensor(labels, dtype=torch.long)
    except OSError as e:
      raise IOError(
          f"Error opening or reading labels file: {labels_path}"
      ) from e

    images_path = os.path.join(self.data_root, "t10k-images-idx3-ubyte")
    try:
      with open(images_path, "rb") as f:
        # Skip header
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(-1, 1, 28, 28)
        self.data = torch.tensor(images, dtype=torch.float32) / 255.0
    except OSError as e:
      raise IOError(
          f"Error opening or reading images file: {images_path}"
      ) from e

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = self.data[index]
    label = self.targets[index]

    # Apply normalization manually (C++ .map(transforms::Normalize))
    # (X - mean) / std_dev
    image = (image - self.normalize_mean) / self.normalize_std

    return image, label


class LenetTest(absltest.TestCase):

  def test_run_test(self):
    with open(MLIR_SRC_PATH, "r") as infile:
      mlir_src = infile.read()

    # Note: The C++ code this was ported from uses a custom stack transform
    # after normalization. In Python, DataLoader handles batching, which
    # effectively stacks the tensors.
    test_dataset = CustomMNISTTestDataset(data_root=DATA_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # SequentialSampler equivalent
    )

    total = 4
    correct = 0
    samples_processed = 0

    for batch_data, batch_target in test_loader:
      if samples_processed >= total:
        break

      input_tensor = batch_data.contiguous()  # (1, 1, 28, 28)
      input_vector = input_tensor.flatten().tolist()
      (output, runtime_ms) = lenet.lenet_interpreter(mlir_src, input_vector)
      print(f"runtime (ms): {runtime_ms}")

      label = batch_target.item()
      max_id = max(range(len(output)), key=lambda index: output[index])

      if max_id == label:
        correct += 1

      print(f"max_id: {max_id}, label: {label}")
      samples_processed += 1

    self.assertGreaterEqual(correct, 0.75 * total)


if __name__ == "__main__":
  absltest.main()
