import sys
from typing import List, Union, Tuple, Optional

sys.path.append("./")
from utility.labml.internal.util.colors import StyleCode


class Destination:
    def log(self, parts: List[Union[str, Tuple[str, Optional[StyleCode]]]], *,
            is_new_line: bool,
            is_reset: bool):
        raise NotImplementedError()
