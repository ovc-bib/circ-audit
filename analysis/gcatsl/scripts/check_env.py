import tensorflow as tf, numpy as np, scipy, sklearn
print("TF", tf.__version__, "| numpy", np.__version__, "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)
import tensorflow.compat.v1 as tf1
print("graph ops:", tf1.Session is not None, "| conv1d:", tf1.layers.conv1d is not None, "| np.bool:", np.bool)
