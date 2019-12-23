# Lint as: python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from quantization import Tensor

# Util functions.
def saturate_rounding_doubling_multiply(value1, value2, datatype):
  # rounding is essentially rounded_value = floor(value + 0.5)
  if datatype == np.int16:
    half = 2 ** 14
    base = 2 ** 15
    min_value = -(2 ** 15)
    max_value = 2 ** 15 - 1
  elif datatype == np.int32:
    half = 2 ** 30
    base = 2 ** 31
    min_value = - (2 ** 31)
    max_value = 2 ** 31 - 1
  else:
    assert False  # Not reached.
  temp = np.int64(value1) * np.int64(value2)
  if temp < 0:
    half *= -1
  temp += half
  temp /= base
  return min(max(temp, min_value), max_value)


def rounding_divide_by_POT(value, exponent):
  assert exponent >= 1
  half = 2 ** (exponent - 1)
  if value < 0:
    half *= -1
  return (value + half) / (2 ** exponent)


def apply_shift(value, exponent):
  if exponent >= 0:
    # Left shift
    value *= (2 ** exponent)
  else:
    # Right shift
    value = rounding_divide_by_POT(value, -exponent)
  return value

def dequant(tensor):
  return (np.array(tensor.value, dtype=np.int32) - tensor.zero_point) * tensor.scale


def diff(value1, value2):
  diff = np.abs(value1 - value2)
  return np.sum(diff) / (value1.size * 1.0)
