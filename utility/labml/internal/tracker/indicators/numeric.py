from abc import ABC
from collections import deque
from typing import Dict, Optional

import numpy as np

from labml.internal.util.values import to_numpy
from . import Indicator


class NumericIndicator(Indicator, ABC):
    def get_mean(self) -> float:
        raise NotImplementedError()

    def get_histogram(self):
        raise NotImplementedError()

    @property
    def mean_key(self):
        return f'{self.name}.mean'


class Queue(NumericIndicator):
    def __init__(self, name: str, queue_size=10, is_print=False, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=is_print, options=options)
        self.queue_size = queue_size
        self._values = deque(maxlen=queue_size)
        self._is_empty = True

    def collect_value(self, value):
        self._is_empty = False
        self._values.append(to_numpy(value).ravel())

    def to_dict(self) -> Dict:
        res = super().to_dict().copy()
        res.update({'queue_size': self._values.maxlen})
        return res

    def is_empty(self) -> bool:
        return len(self._values) == 0 or self._is_empty

    def clear(self):
        self._is_empty = True

    def get_mean(self) -> float:
        return float(np.mean(self._values))

    def get_histogram(self):
        return self._values

    @property
    def mean_key(self):
        return f'{self.name}.mean'

    def copy(self, key: str):
        return Queue(key, queue_size=self.queue_size, is_print=self.is_print, options=self.options)

    def equals(self, value: any) -> bool:
        if not super().equals(value):
            return False
        return value.queue_size == self.queue_size


class _Collection(NumericIndicator, ABC):
    def __init__(self, name: str, is_print: bool, options: Optional[Dict] = None):
        super().__init__(name=name, is_print=is_print, options=options)
        self._values = []

    def _merge(self):
        if len(self._values) == 0:
            return []
        elif len(self._values) == 1:
            return self._values[0]
        else:
            merged = np.concatenate(self._values, axis=0)
            self._values = [merged]

            return merged

    def collect_value(self, value):
        value = to_numpy(value).ravel()
        if len(value) > 0:
            self._values.append(to_numpy(value).ravel())

    def clear(self):
        self._values = []

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def get_mean(self) -> float:
        return float(np.mean(self._merge()))

    def get_histogram(self):
        return self._merge()


class Histogram(_Collection):
    def copy(self, key: str):
        return Histogram(key, is_print=self.is_print, options=self.options)


class Scalar(_Collection):
    def get_histogram(self):
        return None

    def copy(self, key: str):
        return Scalar(key, is_print=self.is_print, options=self.options)

    @property
    def mean_key(self):
        return f'{self.name}'
