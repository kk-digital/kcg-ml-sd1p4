import os
import threading
import time

from labml import monit
from labml.internal.computer import monitor
from labml.internal.computer.configs import computer_singleton
from labml.internal.computer.projects.methods import call_sync
from labml.internal.computer.projects.polling import Polling


def _is_pid_running(pid):
    import os
    import errno

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            return True
        else:
            raise
    else:
        return True


def _get_running_process():
    pid_file = computer_singleton().config_folder / 'monitor.pid'
    if not pid_file.exists():
        return 0

    with open(str(pid_file), 'r') as f:
        pid = f.read()
        try:
            pid = int(pid)
        except ValueError:
            return 0

        if _is_pid_running(pid):
            return pid
        else:
            return 0


class _SyncThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=False)
        self.is_stopped = False
        self.polling = Polling()

    def stop(self):
        self.polling.is_stopped = True

    def run(self):
        call_sync()
        self.polling.run()


def _sync_thread():
    thread = _SyncThread()
    thread.start()


def run(is_check_process: bool = True, open_browser: bool = True):
    pid = _get_running_process()
    if is_check_process and pid:
        raise RuntimeError(f'This computer is already being monitored. PID: {pid}')

    from uuid import uuid1
    session_uuid = uuid1().hex
    with open(str(computer_singleton().config_folder / 'session.txt'), 'w') as f:
        f.write(session_uuid)

    with open(str(computer_singleton().config_folder / 'monitor.pid'), 'w') as f:
        f.write(str(os.getpid()))

    m = monitor.MonitorComputer(session_uuid, open_browser)

    m.start()

    _sync_thread()

    i = 0
    while True:
        with monit.section('Track', is_new_line=False):
            m.track()
        time.sleep(min(60.0, max(1.0, i / 5.0)))
        i += 1


if __name__ == '__main__':
    run()
