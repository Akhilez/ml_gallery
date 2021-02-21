import numpy as np
import tensorflow as tf


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(1, 16),  # TODO: This is pytorch, make it keras like
        tf.keras.layers.Conv2D(16, 32),
    ]
)

model.compile()
