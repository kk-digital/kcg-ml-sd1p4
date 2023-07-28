import copy
from typing import Dict, Optional


class Indicator:
    def __init__(self, *, name: str, is_print: bool, options: Optional[Dict]):
        self.is_print = is_print
        self.name = name
        if options is None:
            options = {}
        else:
            options = copy.deepcopy(options)
        self.options = options

    def clear(self):
        pass

    def is_empty(self) -> bool:
        raise NotImplementedError()

    def to_dict(self) -> Dict:
        return dict(class_name=self.__class__.__name__,
                    name=self.name,
                    is_print=self.is_print)

    def collect_value(self, value):
        raise NotImplementedError()

    def copy(self, key: str):
        raise NotImplementedError()

    def equals(self, value: any) -> bool:
        if type(value) != type(self):
            return False
        return value.name == self.name and value.is_print == self.is_print
