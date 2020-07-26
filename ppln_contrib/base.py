from abc import ABC, abstractmethod
from typing import Any, Dict, List, NoReturn, Optional, Set, TypeVar, Union

import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from ppln.data.transforms import make_albumentations
from ppln.factory import make_model, make_optimizer, make_scheduler
from ppln.hooks import DistSamplerSeedHook, IterTimerHook, LogBufferHook
from ppln.hooks.dist import ApexDDPHook, PytorchBDPHook, PytorchDDPHook, PytorchDPHook
from ppln.hooks.registry import HOOKS
from ppln.parallel.balanced_dp import BalancedDataParallelCriterion
from ppln.utils.config import Config, ConfigDict
from ppln.utils.dist import get_dist_info
from ppln.utils.misc import cached_property, object_from_dict

T = TypeVar("T")

__all__ = ["DPBaseBuilder", "DDPBaseBuilder", "BaseBatchProcessor"]


class _BaseBuilder(ABC):
    dp_hooks: Set[T] = {PytorchDPHook, PytorchBDPHook}
    ddp_hooks: Set[T] = {ApexDDPHook, PytorchDDPHook}

    def __init__(self, config: Config) -> NoReturn:
        self._config = config
        self._hooks = [object_from_dict(hook, HOOKS) for hook in self.config.HOOKS]
        self._extra_hooks = [IterTimerHook(), LogBufferHook()]
        self._has_bdp_hook = self._check_hook_existence(PytorchBDPHook)

    def _check_hook_existence(self, hook: T) -> bool:
        return hook in map(type, self._hooks)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def device(self) -> torch.device:
        device = self.config.get("DEVICE", "cpu")
        return torch.device(device)

    @cached_property
    def model(self) -> nn.Module:
        return make_model(self.config.MODEL, device=self.device)

    @cached_property
    def optimizer(self) -> Optimizer:
        return make_optimizer(self.model, self.config.OPTIMIZER)

    @property
    def scheduler(self) -> Union[object, _LRScheduler]:
        return make_scheduler(self.optimizer, self.config.SCHEDULER)

    @property
    def losses(self) -> Dict[str, T]:
        losses = {k: object_from_dict(v) for k, v in self.config.LOSSES.items()}
        if self._has_bdp_hook:
            return {k: BalancedDataParallelCriterion(v) for k, v in losses.items()}
        return losses

    @property
    def metrics(self) -> Optional[Dict[str, T]]:
        if hasattr(self.config, "METRICS"):
            return {k: object_from_dict(v) for k, v in self.config.METRICS.items()}
        return None

    def transform(self, mode: str) -> A.Compose:
        return make_albumentations(self.config.TRANSFORMS[mode])

    def data_loader(self, mode: str) -> DataLoader:
        dataset = self.dataset(mode)
        return DataLoader(
            dataset=dataset,
            sampler=self.sampler(mode, dataset),
            shuffle=False,
            batch_size=self.config.DATA_LOADER.batch_per_gpu,
            num_workers=self.config.DATA_LOADER.workers_per_gpu,
            pin_memory=self.config.DATA_LOADER.pin_memory,
            drop_last=("train" in mode),
        )

    @property
    @abstractmethod
    def hooks(self) -> List[Union[ConfigDict, T]]:
        ...

    @abstractmethod
    def sampler(self, mode: str, dataset: Dataset) -> Sampler:
        ...

    @abstractmethod
    def dataset(self, mode: str) -> Dataset:
        ...


class DPBaseBuilder(_BaseBuilder):
    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        gpu_count = torch.cuda.device_count()
        for argument_name in self._config.DATA_LOADER.keys():
            if "per_gpu" in argument_name:
                self._config.DATA_LOADER[argument_name] *= gpu_count
        if self._has_bdp_hook:
            self._config.OPTIMIZER.lr /= torch.cuda.device_count()

    @property
    def hooks(self) -> List[Union[ConfigDict, T]]:
        hook_types = set(map(type, self._hooks))
        assert (
            len(hook_types.intersection(self.dp_hooks)) <= 1 and len(hook_types.intersection(self.ddp_hooks)) == 0
        ), (
            f"hooks must only contain something from the list: {', '.join(map(lambda x: x.__name__, self.dp_hooks))}"
            f" or nothing"
        )
        return self._hooks + self._extra_hooks

    def sampler(self, mode: str, dataset: Dataset) -> Sampler:
        if "train" in mode:
            if self.config.DEBUG:
                return WeightedRandomSampler(weights=np.ones(len(dataset)), num_samples=self.config.DEBUG_TRAIN_SIZE)
            else:
                return RandomSampler(dataset, replacement=False)
        else:
            return SequentialSampler(dataset)


class DDPBaseBuilder(_BaseBuilder):
    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__(*args, **kwargs)
        _, world_size = get_dist_info()
        self._config.OPTIMIZER.lr /= world_size

    @property
    def hooks(self) -> List[Union[ConfigDict, T]]:
        hook_types = set(map(type, self._hooks))
        assert (
            len(hook_types.intersection(self.ddp_hooks)) == 1 and len(hook_types.intersection(self.dp_hooks)) == 0
        ), f"hooks must only contain something from the list: {', '.join(map(lambda x: x.__name__, self.ddp_hooks))}"
        return self._hooks + self._extra_hooks + [DistSamplerSeedHook()]

    def sampler(self, mode: str, dataset: Dataset) -> Sampler:
        rank, world_size = get_dist_info()
        if "train" in mode:
            shuffle = True
            if self.config.DEBUG:
                return DistributedSampler(
                    [1] * self.config.DEBUG_TRAIN_SIZE, num_replicas=world_size, rank=rank, shuffle=shuffle
                )
            else:
                return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        else:
            return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)


class BaseBatchProcessor(ABC):
    def __init__(self, builder: _BaseBuilder) -> NoReturn:
        self.config = builder.config
        self.builder = builder

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def val_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def test_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        ...

    def estimate(self, name: str, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        estimators = self.builder.losses
        if self.builder.metrics:
            estimators.update(self.builder.metrics)

        return estimators[name](inputs, targets)
