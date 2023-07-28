from typing import Union, List, Tuple

from labml import logger
from labml.internal.util.colors import StyleCode
from labml.logger import Text


def labml_notice(message: Union[str, List[Union[str, Tuple[str, StyleCode]]]], *, is_danger=False, is_warn=True, is_lite=False):
    if not is_lite:
        log = [('\n' + '-' * 50, Text.subtle)]
        if is_danger:
            log.append(('\nLABML ERROR\n', [Text.danger, Text.title]))
        elif is_warn:
            log.append(('\nLABML WARNING\n', [Text.warning, Text.title]))
        else:
            log.append(('\nLABML MESSAGE\n', [Text.title]))
    else:
        log = ['\n']
        if is_danger:
            log.append(('LABML ERROR: ', [Text.danger, Text.title]))
        elif is_warn:
            log.append(('LABML WARNING: ', [Text.warning, Text.title]))
        else:
            log.append(('LABML MESSAGE: ', [Text.title]))

    if isinstance(message, str):
        log.append(message)
    else:
        log += message

    if not is_lite:
        log.append(('\n' + '-' * 50, Text.subtle))

    logger.log(log)
