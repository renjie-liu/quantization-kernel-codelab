# Lint as: python3

import numpy as np
from quantization import Tensor

# Data prepare.
def populate_data(shape, min_value, max_value):
  value_range = max_value - min_value
  float_value = np.random.random_sample(shape) * value_range + min_value
  scale = (value_range) / 256
  zero_point = -int((max_value + min_value) / (2.0 * scale))
  quant_value = np.array(float_value / float(scale) + zero_point, dtype=np.int8)
  q_tensor = Tensor(quant_value, scale, zero_point)
  return float_value, q_tensor
