from typing import Optional, Union, Tuple
import sys

sys.path.append("./")
from utility.labml.internal.util.colors import StyleCode

LogPart = Union[str, Tuple[str, Optional[StyleCode]]]
