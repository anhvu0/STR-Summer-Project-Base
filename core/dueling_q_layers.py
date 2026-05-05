try:
    from keras import ops
except ImportError:  # pragma: no cover - compatibility for older keras builds
    from keras import backend as ops
from keras.layers import Layer
try:
    from keras.saving import register_keras_serializable
except ImportError:  # pragma: no cover - compatibility for older keras builds
    from keras.utils import register_keras_serializable


@register_keras_serializable(package="STRProject")
class DuelingQCombine(Layer):
    """
    Combine scalar state-value and per-action advantages into centered Q-values.
    """

    def call(self, inputs):
        value, advantage = inputs
        centered_advantage = advantage - ops.mean(advantage, axis=-1, keepdims=True)
        return value + centered_advantage

    def compute_output_shape(self, input_shape):
        _, advantage_shape = input_shape
        return advantage_shape

    def get_config(self):
        return super().get_config()
