import collections
import math
from typing import TYPE_CHECKING, List, Tuple

from labml.internal.util.colors import StyleCode
from ....logger import Text

if TYPE_CHECKING:
    from . import Logger


def _format_bool(value: bool):
    return 'true' if value else 'false'


def _format_int(value: int):
    return f"{value:,}"


def _format_float(value: float):
    if math.isnan(value):
        return ' ' * 5 + 'nan'

    if abs(value) < 1e-9:
        lg = 0
    else:
        lg = int(math.ceil(math.log10(abs(value)))) + 1

    decimals = 7 - lg
    decimals = max(1, decimals)
    decimals = min(6, decimals)

    fmt = "{v:8,." + str(decimals) + "f}"
    return fmt.format(v=value)


def _format_value(value: any):
    if isinstance(value, bool):
        return _format_bool(value)
    elif isinstance(value, int):
        return _format_int(value)
    elif isinstance(value, float):
        return _format_float(value)

    try:
        import numpy
    except ImportError:
        numpy = None

    if numpy is None:
        return None

    if isinstance(value, numpy.number) and numpy.issubdtype(value, numpy.integer):
        return _format_int(int(value))
    elif isinstance(value, numpy.number) and numpy.issubdtype(value, numpy.floating):
        if numpy.isnan(value):
            return 'NaN'
        return _format_float(float(value))

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        if isinstance(value, torch.Tensor):
            if not value.shape:
                return _format_value(value.item())
            else:
                return None

    return None


def _format_tensor(s: List[str], limit: int = 1_000, style: StyleCode = Text.value):
    res = []
    length = 0
    for p in s:
        if p in [',', ']', '[', '...']:
            res.append((p, Text.subtle))
        else:
            res.append((p, style))
            length += len(p)

        if length > limit:
            res.append((' ... ', Text.warning))
            return res, True

    return res, False


def _key_value_pair(key: any, value: any, style: StyleCode = Text.meta):
    f = _format_value(value)
    if f is None:
        f = str(value)
    return [(f'{str(key)}: ', Text.subtle),
            (f, style)]


def _render_tensor(tensor, *, new_line: str = '\n', indent: str = '', depth=0):
    if len(tensor) > 5:
        truncated = True
        idx = [0, 1, 2, '...', len(tensor) - 1]
    else:
        truncated = False
        idx = [i for i in range(len(tensor))]

    res = [indent, '[']
    if depth >= 2:
        new_line = ''
    else:
        new_line = new_line

    if new_line == '\n':
        current_indent = ' ' + indent
    else:
        indent = ''
        current_indent = ''
    if len(tensor.shape) > 1:
        res.append(new_line)
        for i in idx:
            if i == '...':
                res.append(current_indent)
                res.append('...')
            else:
                sub_res, sub_trunc = _render_tensor(tensor[i],
                                                    new_line=new_line,
                                                    indent=current_indent,
                                                    depth=depth + 1,
                                                    )
                truncated = truncated or sub_trunc
                res += sub_res
            if i != idx[-1]:
                res.append(', ')
            res.append(new_line)
    else:
        for i in idx:
            if i == '...':
                res.append('...')
            else:
                res += [_format_value(tensor[i])]
            if i != idx[-1]:
                res.append(', ')

    res.append(indent)
    res.append(']')

    return res, truncated


def _get_value_full(value: any):
    if isinstance(value, str):
        if len(value) < 500:
            return [('"', Text.subtle),
                    (value, Text.value),
                    ('" len(', Text.subtle),
                    (_format_int(len(value)), Text.meta),
                    (')', Text.subtle)]
        else:
            return [('"', Text.subtle),
                    (value[:500], Text.value),
                    (' ..." len(', Text.subtle),
                    (_format_int(len(value)), Text.meta),
                    (')', Text.subtle)]

    try:
        import numpy
    except ImportError:
        numpy = None

    if numpy is not None and isinstance(value, numpy.ndarray):
        arr, trunc = _render_tensor(value, new_line='\n')
        arr, trunc_format = _format_tensor(arr)
        if not trunc and not trunc_format:
            return arr
        else:
            return [*_key_value_pair('dtype', value.dtype),
                    '\n',
                    *_key_value_pair('shape', [s for s in value.shape], Text.value),
                    '\n',
                    *_key_value_pair('min', numpy.min(value)),
                    ' ',
                    *_key_value_pair('max', numpy.max(value)),
                    ' ',
                    *_key_value_pair('mean', numpy.mean(value)),
                    ' ',
                    *_key_value_pair('std', numpy.std(value)),
                    '\n',
                    *arr]

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        arr, trunc = _render_tensor(value, new_line='\n')
        arr, trunc_format = _format_tensor(arr)
        if not trunc and not trunc_format:
            return arr
        else:
            return [*_key_value_pair('dtype', value.dtype),
                    '\n',
                    *_key_value_pair('shape', [s for s in value.shape], Text.value),
                    '\n',
                    *_key_value_pair('min', torch.min(value).item()),
                    ' ',
                    *_key_value_pair('max', torch.max(value).item()),
                    ' ',
                    *_key_value_pair('mean', torch.mean(value.to(torch.float)).item()),
                    ' ',
                    *_key_value_pair('std', torch.std(value.to(torch.float)).item()),
                    '\n',
                    *arr]

    s = str(value)
    s = s.replace('\r', '')
    return [s]


def _shrink(s: str, style: StyleCode = Text.value, limit: int = 80):
    s = s.replace('\r', '')
    lines = s.split('\n')

    res = []
    length = 0
    for line in lines:
        if len(res) > 0:
            res.append(('\\n', Text.subtle))
        if len(line) + length < limit:
            res.append((line, style))
            length += len(line)
        else:
            res.append((line[:limit - length], style))
            res.append((' ...', Text.warning))
            break

    return res


def _get_value_line(value: any):
    f = _format_value(value)

    if f is not None:
        return [(f, Text.value)]
    elif isinstance(value, str):
        return [('"', Text.subtle)] + _shrink(value) + [('"', Text.subtle)]

    try:
        import numpy
    except ImportError:
        numpy = None

    if numpy is not None and isinstance(value, numpy.ndarray):
        return [*_format_tensor(_render_tensor(value, new_line='')[0], limit=80)[0]]

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return [*_format_tensor(_render_tensor(value, new_line='')[0], limit=80)[0]]

    s = str(value)
    return _shrink(s)


class _InspectLogger:
    def __init__(self):
        self.parts = []

    def _log_key_value(self, items: List[Tuple[any, any]],
                       is_show_count: bool = True,
                       is_expand: bool = False,
                       n_key_values: int = 10):
        max_key_len = 0
        for k, v in items:
            max_key_len = max(max_key_len, len(str(k)))

        count = len(items)
        if n_key_values > 0 and count > n_key_values + 1:
            items = items[:n_key_values]

        if is_expand:
            for k, v in items:
                self.parts.append([(k, Text.heading)])
                self.parts.append(_get_value_full(v))
        else:
            for k, v in items:
                spaces = " " * (max_key_len - len(str(k)))
                self.parts.append([(f"{spaces}{k}: ", Text.key)] +
                                  _get_value_line(v))

        if n_key_values > 0 and count > n_key_values + 1:
            self.parts.append([("...", Text.meta)])

        if is_show_count:
            self.parts.append([
                "Total ",
                (str(count), Text.meta),
                " item(s)"])

    def info(self, *args, **kwargs):
        if '_n' in kwargs:
            n_key_values = kwargs['_n']
            del kwargs['_n']
        else:
            n_key_values = 10

        if '_expand' in kwargs:
            is_expand = kwargs['_expand']
            del kwargs['_expand']
        else:
            is_expand = None

        if len(args) == 0 and len(kwargs) == 1 and is_expand != True:
            self.parts.append([(next(iter(kwargs.keys())), [Text.heading, Text.key])])
            args = [next(iter(kwargs.values()))]
            kwargs = {}

        if is_expand is None:
            is_expand = False

        if len(args) == 0:
            self._log_key_value([(k, v) for k, v in kwargs.items()],
                                is_show_count=False,
                                n_key_values=n_key_values,
                                is_expand=is_expand)
        elif len(args) == 1:
            assert len(kwargs.keys()) == 0
            arg = args[0]
            if isinstance(arg, collections.UserList):
                arg = list(*arg)

            if isinstance(arg, list) or isinstance(arg, tuple):
                self._log_key_value([(i, v) for i, v in enumerate(arg)],
                                    n_key_values=n_key_values,
                                    is_expand=is_expand)
                return

            if isinstance(arg, collections.UserDict):
                arg = dict(**arg)
            if isinstance(arg, dict):
                keys = list(arg.keys())
                keys.sort(key=lambda x: str(x))
                self._log_key_value([(k, arg[k]) for k in keys],
                                    n_key_values=n_key_values,
                                    is_expand=is_expand)

                return

            from labml.internal.analytics.models import ValueCollection
            if isinstance(arg, ValueCollection):
                self._log_key_value([(i, v) for i, v in enumerate(arg.keys())],
                                    n_key_values=n_key_values,
                                    is_expand=is_expand)
                return

            self.parts.append(_get_value_full(arg))
        else:
            assert len(kwargs.keys()) == 0
            self._log_key_value([(i, v) for i, v in enumerate(args)],
                                is_show_count=False,
                                n_key_values=n_key_values,
                                is_expand=is_expand)


class Inspect:
    def __init__(self, logger: 'Logger'):
        self.__logger = logger

    def info(self, *args, **kwargs):
        ins = _InspectLogger()
        ins.info(*args, **kwargs)

        parts = []

        for ps in ins.parts:
            parts += ['\n'] + ps

        if parts:
            self.__logger.log(parts[1:])
