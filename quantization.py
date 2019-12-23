# Lint as: python3

class Tensor:
  def __init__(self, value, scale, zero_point):
    self.value = value
    self.scale = scale
    self.zero_point = zero_point
    # There're other variants need to consider:
    # Datatype? Bits? Per-axis?
