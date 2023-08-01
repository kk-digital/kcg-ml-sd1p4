import os, os.path
from  pathlib import Path
from typing import Set, List


def get_caller_file(ignore_callers: Set[str] = None) -> str:
    if ignore_callers is None:
        ignore_callers = {}

    import inspect

    frames: List[inspect.FrameInfo] = inspect.stack()
    lab_src = Path(__file__).absolute().parent.parent

    for f in frames:
        module_path = Path(f.filename)
        if str(module_path).startswith(os.path.join(lab_src, '')):
            continue
        if '/labml_helpers/' in str(module_path):
            continue
        if str(module_path) in ignore_callers:
            continue
        if str(module_path).startswith('<stdin'):
            break
        if str(module_path).startswith('<ipython'):
            break
        if not module_path.exists():
            break
        if str(module_path.absolute()).find('/dist-packages/') != -1:
            continue
        return str(module_path)

    return str(Path('').absolute())
