import json
import tarfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from labml import logger, lab, monit
from labml.internal.configs.processor import load_configs
from labml.utils.download import download_file
from .. import util
from ..manage.runs import get_run_by_uuid, get_checkpoints
from ...logger import Text
from ...utils.notice import labml_notice


def struct_time_to_time(t: time.struct_time):
    return f"{t.tm_hour :02}:{t.tm_min :02}:{t.tm_sec :02}"


def struct_time_to_date(t: time.struct_time):
    return f"{t.tm_year :04}-{t.tm_mon :02}-{t.tm_mday :02}"


_GLOBAL_STEP = 'global_step'
_TAR_BUFFER_SIZE = 100_000


class RunLoadError(TypeError):
    pass


class RunInfo:
    def __init__(self, *,
                 uuid: str,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 name: str,
                 comment: str,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 repo_remotes: List[str] = None,
                 is_dirty: bool = True,
                 experiment_path: Path,
                 start_step: int = 0,
                 notes: str = '',
                 load_run: Optional[str] = None,
                 tags: List[str]):
        self.name = name
        self.uuid = uuid
        if repo_remotes is None:
            repo_remotes = []
        self.repo_remotes = repo_remotes
        self.commit = commit
        self.is_dirty = is_dirty
        self.python_file = python_file
        self.trial_date = trial_date
        self.trial_time = trial_time
        self.comment = comment
        self.commit_message = commit_message
        self.start_step = start_step

        self.load_run = load_run

        self.experiment_path = experiment_path
        self.run_path = experiment_path / str(uuid)
        self.pids_path = self.run_path / 'pids'
        self.checkpoint_path = self.run_path / "checkpoints"
        self.numpy_path = self.run_path / "numpy"

        self.diff_path = self.run_path / "source.diff"

        self.sqlite_path = self.run_path / "sqlite.db"
        self.artifacts_folder = self.run_path / "artifacts"
        self.tensorboard_log_path = self.run_path / "tensorboard"
        self.log_file = self.run_path / 'log.jsonl'

        self.info_path = self.run_path / "run.yaml"
        self.indicators_path = self.run_path / "indicators.yaml"
        self.configs_path = self.run_path / "configs.yaml"
        self.run_log_path = self.run_path / "run_log.jsonl"
        self.notes = notes
        self.tags = tags

    @classmethod
    def from_path(cls, run_path: Path):
        """
        ## Create a new from path
        """
        run_info_path = run_path / 'run.yaml'

        try:
            with open(str(run_info_path), 'r') as f:
                data = util.yaml_load(f.read())
                if not isinstance(data, dict):
                    raise RunLoadError()
                return cls.from_dict(run_path.parent, data)
        except FileNotFoundError:
            raise RunLoadError()

    @classmethod
    def from_dict(cls, experiment_path: Path, data: Dict[str, any]):
        """
        ## Create a new trial from a dictionary
        """
        params = dict(experiment_path=experiment_path)
        params.update(data)
        try:
            run = cls(**params)
            return run
        except TypeError as e:
            raise RunLoadError(e)

    def to_dict(self):
        """
        ## Convert trial to a dictionary for saving
        """
        return dict(
            name=self.name,
            uuid=self.uuid,
            python_file=self.python_file,
            trial_date=self.trial_date,
            trial_time=self.trial_time,
            comment=self.comment,
            repo_remotes=self.repo_remotes,
            commit=self.commit,
            commit_message=self.commit_message,
            is_dirty=self.is_dirty,
            start_step=self.start_step,
            notes=self.notes,
            tags=self.tags,
            load_run=self.load_run
        )

    def pretty_print(self) -> List[str]:
        """
        ## 🎨 Pretty print trial for the python file header
        """

        # Trial information
        commit_status = "[dirty]" if self.is_dirty else "[clean]"
        res = [
            f"{self.trial_date} {self.trial_time}",
            self.comment,
            f"[{commit_status}]: {self.commit_message}",
            f"start_step: {self.start_step}"
        ]
        return res

    def __str__(self):
        return f"{self.__class__.__name__}(comment=\"{self.comment}\"," \
               f" commit=\"{self.commit_message}\"," \
               f" date={self.trial_date}, time={self.trial_time})"

    def __repr__(self):
        return self.__str__()

    def is_after(self, run: 'Run'):
        if run.trial_date < self.trial_date:
            return True
        elif run.trial_date > self.trial_date:
            return False
        elif run.trial_time < self.trial_time:
            return True
        else:
            return False


class Run(RunInfo):
    """
    # Trial 🏃‍

    Every trial in an experiment has same configs.
    It's just multiple runs.

    A new trial will replace checkpoints and TensorBoard summaries
    or previous trials, you should make a copy if needed.
    The performance log in `trials.yaml` is not replaced.

    You should run new trials after bug fixes or to see performance is
     consistent.

    If you want to try different configs, create multiple experiments.
    """

    diff: Optional[str]

    def __init__(self, *,
                 uuid: str,
                 python_file: str,
                 trial_date: str,
                 trial_time: str,
                 name: str,
                 comment: str,
                 repo_remotes: List[str] = None,
                 commit: Optional[str] = None,
                 commit_message: Optional[str] = None,
                 is_dirty: bool = True,
                 experiment_path: Path,
                 start_step: int = 0,
                 notes: str = '',
                 tags: List[str]):
        super().__init__(python_file=python_file, trial_date=trial_date, trial_time=trial_time,
                         name=name, comment=comment, uuid=uuid, experiment_path=experiment_path,
                         repo_remotes=repo_remotes,
                         commit=commit, commit_message=commit_message, is_dirty=is_dirty,
                         start_step=start_step, notes=notes, tags=tags)
        self.diff = None

    @classmethod
    def create(cls, *,
               uuid: str,
               experiment_path: Path,
               python_file: str,
               trial_time: time.struct_time,
               name: str,
               comment: str,
               tags: List[str]):
        """
        ## Create a new trial
        """
        return cls(python_file=python_file,
                   trial_date=struct_time_to_date(trial_time),
                   trial_time=struct_time_to_time(trial_time),
                   uuid=uuid,
                   experiment_path=experiment_path,
                   name=name,
                   comment=comment,
                   tags=tags)

    def make_path(self):
        run_path = Path(self.run_path)
        if not run_path.exists():
            try:
                run_path.mkdir(parents=True)
            except FileExistsError:
                pass

    def save_info(self):
        self.make_path()

        with open(str(self.info_path), "w") as file:
            file.write(util.yaml_dump(self.to_dict()))

        if self.diff is not None:
            with open(str(self.diff_path), "w") as f:
                f.write(self.diff)


def _get_run_checkpoint(run_path: Path, checkpoint: int = -1):
    checkpoints = list(get_checkpoints(run_path))
    if not checkpoints:
        return None

    checkpoint_steps = [int(c.name) for c in checkpoints]
    if checkpoint < 0:
        required_ci = np.max(checkpoint_steps) + checkpoint + 1
    else:
        required_ci = checkpoint

    selected_ci = -1
    for ci in checkpoint_steps:
        if ci <= required_ci:
            selected_ci = max(selected_ci, ci)

    return selected_ci


def get_run_checkpoint(run_uuid: str, checkpoint: int = -1):
    run_path = get_run_by_uuid(lab.get_experiments_path(), run_uuid)
    if run_path is None:
        logger.log("Couldn't find a previous run")
        return None, None

    checkpoint = _get_run_checkpoint(run_path, checkpoint)

    if checkpoint is None:
        logger.log("Couldn't find checkpoints")
        return None, None

    logger.log(["Selected ",
                ("experiment", Text.key),
                " = ",
                (run_path.parent.name, Text.value),
                " ",
                ("run", Text.key),
                " = ",
                (run_uuid, Text.value),
                " ",
                ("checkpoint", Text.key),
                " = ",
                (f'{checkpoint :,}', Text.value)])

    checkpoint_path = run_path / "checkpoints"
    return checkpoint_path / str(checkpoint), checkpoint


def get_configs(run_uuid: str):
    run_path = get_run_by_uuid(lab.get_experiments_path(), run_uuid)
    if run_path is None:
        labml_notice(["Couldn't find a previous run to load configurations: ",
                      (run_uuid, Text.value)], is_danger=True)
        return None

    configs_path = run_path / "configs.yaml"
    configs = load_configs(configs_path)

    return configs


def save_bundle(path: Path, run_uuid: str, checkpoint: int = -1, *,
                data_files: List[str]):
    run_path = get_run_by_uuid(lab.get_experiments_path(), run_uuid)
    if run_path is None:
        raise RuntimeError(f"Couldn't find run {run_uuid}")

    checkpoint = _get_run_checkpoint(run_path, checkpoint)

    if checkpoint is None:
        raise RuntimeError(f"Couldn't find checkpoint {run_uuid}:{checkpoint}")

    info_path = path.parent / f'{path.stem}.info.json'
    info = {'uuid': run_uuid, 'checkpoint': checkpoint}
    with open(str(info_path), 'w') as f:
        f.write(json.dumps(info))

    checkpoint_path = run_path / "checkpoints" / str(checkpoint)

    with monit.section('Create bundle'):
        with tarfile.open(str(path), 'w:gz') as tar:
            tar.add(str(checkpoint_path), 'checkpoint')
            tar.add(str(run_path / 'run.yaml'), 'run.yaml')
            tar.add(str(run_path / 'configs.yaml'), 'configs.yaml')
            tar.add(str(info_path), 'info.json')
            for f in data_files:
                tar.add(str(lab.get_data_path() / f), f'data/{f}')

    info_path.unlink()


def _extract_tar_file(tar: tarfile.TarFile, f: tarfile.TarInfo, path: Path):
    with tar.extractfile(f) as ef:
        with open(str(path), 'wb') as o:
            while True:
                r = ef.read(_TAR_BUFFER_SIZE)
                if r:
                    o.write(r)
                else:
                    break


def load_bundle(path: Path, url: Optional[str] = None) -> Tuple[str, int]:
    if url:
        download_file(url, path)

    if not path.exists():
        raise FileNotFoundError(f'Bundle archive missing: {path}')

    with monit.section('Extract bundle'):
        with tarfile.open(str(path), 'r:gz') as tar:
            files = tar.getmembers()
            info_member = None
            for f in files:
                if f.name == 'info.json':
                    info_member = f

            if not info_member:
                raise RuntimeError(f"Corrupted bundle. Missing info.json")

            with tar.extractfile(info_member) as ef:
                info = json.load(ef)

            run_uuid, checkpoint = info['uuid'], info['checkpoint']
            run_path = get_run_by_uuid(lab.get_experiments_path(), run_uuid)

            if run_path is not None:
                logger.log(f"Run {run_uuid} exists", Text.meta)
                current_checkpoint = _get_run_checkpoint(run_path, checkpoint)
                if checkpoint == current_checkpoint:
                    logger.log(f"Checkpoint {checkpoint} exists", Text.meta)
                    return run_uuid, checkpoint

            run_path = lab.get_experiments_path() / 'bundled' / run_uuid

            checkpoint_path = run_path / "checkpoints" / str(checkpoint)
            if not checkpoint_path.exists():
                checkpoint_path.mkdir(parents=True)

            data_path = lab.get_data_path()
            if not data_path.exists():
                data_path.mkdir(parents=True)

            for f in files:
                if f.name == 'run.yaml':
                    _extract_tar_file(tar, f, run_path / 'run.yaml')
                elif f.name == 'configs.yaml':
                    _extract_tar_file(tar, f, run_path / 'configs.yaml')
                elif f.name.startswith('checkpoint/'):
                    p = f.name[len('checkpoint/'):]
                    p = checkpoint_path / p
                    if not p.parent.exists():
                        p.parent.mkdir(parents=True)
                    _extract_tar_file(tar, f, p)
                elif f.name.startswith('data/'):
                    p = f.name[len('data/'):]
                    p = data_path / p
                    if not p.parent.exists():
                        p.parent.mkdir(parents=True)
                    _extract_tar_file(tar, f, p)

            return run_uuid, checkpoint
