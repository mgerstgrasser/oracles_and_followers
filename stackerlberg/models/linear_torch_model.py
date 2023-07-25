import numpy as np

# from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork as MyVisionNetwork
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_initializer
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


class LinearTorchModel(TorchModelV2, nn.Module):
    """Linear Torch Model without bias."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._model = nn.Linear(int(np.product(obs_space.shape)), num_outputs, bias=False)
        self._value = nn.Linear(int(np.product(obs_space.shape)), 1, bias=False)
        if model_config.get("custom_model_config", {}).get("constant_init", False):
            torch.nn.init.constant_(self._model.weight, 0.01)
            torch.nn.init.constant_(self._value.weight, 0.01)
        else:
            initializer = normc_initializer(0.01)
            initializer(self._model.weight)
            initializer = normc_initializer(0.01)
            initializer(self._value.weight)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._obs = obs
        return self._model(obs), state

    @override(TorchModelV2)
    def value_function(self):
        value = self._value(self._obs)
        return value.squeeze(1)
