from typing import Tuple, Optional, List, overload, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from labml.internal.analytics import cache as _cache
from labml.internal.analytics.altair import density as _density
from labml.internal.analytics.altair import histogram as _histogram
from labml.internal.analytics.altair import scatter as _scatter
from labml.internal.analytics.altair import binned_heatmap as _binned_heatmap
from labml.internal.analytics.indicators import IndicatorCollection as _IndicatorCollection
from labml.internal.analytics.models import ModelProbe as _ModelProbe
from labml.internal.analytics.models import ValueCollection as _ValueCollection
from labml.internal.analytics.models import DeepValueCollection as _DeepValueCollection
from labml.internal.analytics.viz import text_attention as _text_attention
from labml.internal.analytics.viz import init_inline_viz as _init_inline_viz


def _remove_names_prefix(names: List[Union[str, List[str]]]) -> List[str]:
    if len(names) == 0:
        return []

    if isinstance(names[0], list):
        common = names[0]
    else:
        common = None

    for n in names:
        if common is None:
            break
        if not isinstance(n, list):
            common = None
        merge = []
        for x, y in zip(common, n):
            if x != y:
                merge.append(None)
            else:
                merge.append(x)
        common = merge

    res = []
    for n in names:
        if isinstance(n, list):
            if common is not None:
                n = [p for i, p in enumerate(n) if i > len(common) or p != common[i]]
            n = '-'.join(n)

        res.append(n)

    return res


class IndicatorCollection(_IndicatorCollection):
    r"""
    You can get a indicator collection with :func:`runs`.

    >>> from labml import analytics
    >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')

    You can reference individual indicators as attributes.

    >>> train_loss = indicators.train_loss

    You can add multiple indicator collections

    >>> losses = indicators.train_loss + indicators.validation_loss
    """
    pass


class ModelProbe(_ModelProbe):
    r"""
    You can wrap any PyTorch model with :class:`ModelProbe` to access it's parameters, activations and gradients.

    `Here's a notebook with example usage <https://github.com/labmlai/labml/blob/master/guides/model_probe.ipynb>`_

    >>> from labml.analytics import ModelProbe
    >>> probe = ModelProbe(model)
    >>> outputs = model(inputs)
    >>> outputs.backward()
    >>> probe.forward_output["*attention*"].get_dict()
    >>> probe.parameters['*.bias'].get_list()
    """
    pass


class ValueCollection(_ValueCollection):
    pass


class DeepValueCollection(_DeepValueCollection):
    pass


def text_attention(attn: 'torch.Tensor', src_tokens: List[str], tgt_tokens: List[str]):
    _text_attention(attn, src_tokens, tgt_tokens)


def init_inline_viz():
    _init_inline_viz()


def runs(*uuids: str):
    r"""
    This is used to analyze runs.
    It fetches all the log indicators.

    Arguments:
        uuids (str): UUIDs of the runs. You can
            get this from `labml.ai app <https://github.com/labmlai/labml/tree/master/app>`_

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
    """

    indicators = None
    for r in uuids:
        run = _cache.get_run(r)
        indicators = indicators + run.indicators

    return indicators


def get_experiment_runs(experiment_name: str):
    return _cache.get_experiment_runs(experiment_name)


def get_run(uuid: str):
    r"""
    Returns ``Run`` object
    """
    return _cache.get_run(uuid)


def set_preferred_db(db: str):
    """
    Set the preference to load data from.

    Arguments:
        db (str): Either ``tensorboard`` or ``sqlite``
    """
    if db not in ['tensorboard', 'sqlite']:
        raise ValueError('db should be one of tensorboard or sqlite')
    _cache.set_preferred_db(db)


@overload
def distribution(indicators: IndicatorCollection, *,
                 names: Optional[List[str]] = None,
                 levels: int = 5, alpha: int = 0.6,
                 color_scheme: str = 'tableau10',
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def distribution(series: List[Union[np.ndarray, 'torch.Tensor']], *,
                 names: Optional[List[str]] = None,
                 levels: int = 5, alpha: int = 0.6,
                 color_scheme: str = 'tableau10',
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def distribution(series: List[Union[np.ndarray, 'torch.Tensor']],
                 step: np.ndarray, *,
                 names: Optional[List[str]] = None,
                 levels: int = 5, alpha: int = 0.6,
                 color_scheme: str = 'tableau10',
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def distribution(series: Union[np.ndarray, 'torch.Tensor'], *,
                 names: Optional[List[str]] = None,
                 levels: int = 5, alpha: int = 0.6,
                 color_scheme: str = 'tableau10',
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


def distribution(*args: any,
                 names: Optional[List[str]] = None,
                 levels: int = 5, alpha: int = 0.6,
                 color_scheme: str = 'tableau10',
                 height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a distribution plot distribution with Altair

    This has multiple overloads

    .. function:: distribution(indicators: IndicatorCollection, *, names: Optional[List[str]] = None, levels: int = 5, alpha: int = 0.6, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    .. function:: distribution(series: Union[np.ndarray, torch.Tensor], *, names: Optional[List[str]] = None, levels: int = 5, alpha: int = 0.6, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    .. function:: distribution(series: List[Union[np.ndarray, torch.Tensor]], *, names: Optional[List[str]] = None, levels: int = 5, alpha: int = 0.6, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    .. function:: distribution(series: List[Union[np.ndarray, torch.Tensor]], step: np.ndarray, *, names: Optional[List[str]] = None, levels: int = 5, alpha: int = 0.6, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        series(List[np.ndarray]): List of series of data
        step(np.ndarray): Steps

    Keyword Arguments:
        names(List[str]): List of names of series
        levels: how many levels of the distribution to be plotted
        alpha: opacity of the distribution
        color_scheme: color scheme
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.distribution(indicators)
    """

    series = None
    step = None

    if len(args) == 1:
        if isinstance(args[0], _IndicatorCollection):
            series, names_ = _cache.get_indicators_data(args[0])
            if not series:
                raise ValueError("No series found")
            if names is None:
                names = names_
        elif isinstance(args[0], list):
            series = args[0]
        else:
            series = [args[0]]
    elif len(args) == 2:
        series = args[0]
        step = args[1]

    if names is None:
        digits = len(str(len(series)))
        names = [str(i + 1).zfill(digits) for i in range(len(series))]

    if series is None:
        raise ValueError("distribution should be called with an indicator collection"
                         " or a series. Check documentation for details.")

    names = _remove_names_prefix(names)
    tables = _density.data_to_table(series, names, step)

    return _density.render(
        tables,
        levels=levels,
        alpha=alpha,
        color_scheme=color_scheme,
        width=width,
        height=height,
        height_minimap=height_minimap)


def histogram(series: Union[np.ndarray, 'torch.Tensor'], *,
              low: Optional[float] = None, high: Optional[float] = None,
              height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a histogram with Altair

    Arguments:
        series(Union[np.ndarray, torch.Tensor]): Data

    Keyword Arguments:
        low: values less than this are ignored
        high: values greater than this are ignored
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization
    """

    table = _histogram.data_to_table(series, low, high)

    return _histogram.render(
        table,
        width=width,
        height=height,
        height_minimap=height_minimap)


@overload
def scatter(indicators: IndicatorCollection, x_indicators: IndicatorCollection, *,
            names: Optional[List[str]] = None, x_name: Optional[str] = None,
            noise: Optional[Tuple[float, float]] = None,
            circle_size: int = 20,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def scatter(series: List[np.ndarray],
            x_series: np.ndarray, *,
            names: Optional[List[str]] = None, x_name: Optional[str] = None,
            noise: Optional[Tuple[float, float]] = None,
            circle_size: int = 20,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


def scatter(*args: any,
            names: Optional[List[str]] = None, x_name: Optional[str] = None,
            noise: Optional[Tuple[float, float]] = None,
            circle_size: int = 20,
            height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a scatter plot with Altair

    This has multiple overloads

    .. function:: scatter(indicators: IndicatorCollection, x_indicators: IndicatorCollection, *, names: Optional[List[str]] = None, x_name: Optional[str] = None, noise: Optional[Tuple[float, float]] = None, circle_size: int = 20, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    .. function:: scatter(series: List[np.ndarray], x_series: np.ndarray, *, names: Optional[List[str]] = None, x_name: Optional[str] = None, noise: Optional[Tuple[float, float]] = None, circle_size: int = 20, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        x_indicators(IndicatorCollection): Indicator for x-axis
        series(List[np.ndarray]): List of series of data
        x_series(np.ndarray): X series of data

    Keyword Arguments:
        names(List[str]): List of names of series
        name(str): Name of X series
        noise: Noise to be added to spread out the scatter plot
        circle_size: size of circles in the plot
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization

    :Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.scatter(indicators.validation_loss, indicators.train_loss)
    """

    series = None
    x_series = None

    if len(args) == 2:
        if isinstance(args[0], _IndicatorCollection) and isinstance(args[1], _IndicatorCollection):
            series, names_ = _cache.get_indicators_data(args[0])
            x_series, x_name_ = _cache.get_indicators_data(args[1])

            if len(x_series) != 1:
                raise ValueError("There should be exactly one series for x-axis")
            if not series:
                raise ValueError("No series found")
            x_series = x_series[0]
            if x_name is None:
                x_name = x_name_[0]
            if names is None:
                names = names_
        elif isinstance(args[0], list):
            series = args[0]
            x_series = args[1]

    if series is None:
        raise ValueError("scatter should be called with an indicator collection"
                         " or a series. Check documentation for details.")

    if x_name is None:
        x_name = 'x'
    if names is None:
        names = [f'{i + 1}' for i in range(len(series))]

    tables = [_scatter.data_to_table(s, x_series, noise) for s in series]
    names = _remove_names_prefix(names)

    return _scatter.render(
        tables,
        names=names,
        x_name=x_name,
        width=width,
        height=height,
        height_minimap=height_minimap,
        circle_size=circle_size)


@overload
def binned_heatmap(indicators: IndicatorCollection, x_indicators: IndicatorCollection, *,
                   names: Optional[List[str]] = None, x_name: Optional[str] = None,
                   height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


@overload
def binned_heatmap(series: List[np.ndarray],
                   x_series: np.ndarray, *,
                   names: Optional[List[str]] = None, x_name: Optional[str] = None,
                   height: int = 400, width: int = 800, height_minimap: int = 100):
    ...


def binned_heatmap(*args: any,
                   names: Optional[List[str]] = None, x_name: Optional[str] = None,
                   height: int = 400, width: int = 800, height_minimap: int = 100):
    r"""
    Creates a scatter plot with Altair

    This has multiple overloads

    .. function:: binned_heatmap(indicators: IndicatorCollection, x_indicators: IndicatorCollection, *, names: Optional[List[str]] = None, x_name: Optional[str] = None, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    .. function:: binned_heatmap(series: List[np.ndarray], x_series: np.ndarray, *, names: Optional[List[str]] = None, x_name: Optional[str] = None, height: int = 400, width: int = 800, height_minimap: int = 100)
        :noindex:

    Arguments:
        indicators(IndicatorCollection): Set of indicators to be plotted
        x_indicators(IndicatorCollection): Indicator for x-axis
        series(List[np.ndarray]): List of series of data
        x_series(np.ndarray): X series of data

    Keyword Arguments:
        names(List[str]): List of names of series
        name(str): Name of X series
        noise: Noise to be added to spread out the scatter plot
        circle_size: size of circles in the plot
        height: height of the visualization
        width: width of the visualization
        height_minimap: height of the view finder

    Return:
        The Altair visualization

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.scatter(indicators.validation_loss, indicators.train_loss)
    """

    series = None
    x_series = None

    if len(args) == 2:
        if isinstance(args[0], _IndicatorCollection) and isinstance(args[1], _IndicatorCollection):
            series, names_ = _cache.get_indicators_data(args[0])
            x_series, x_name_ = _cache.get_indicators_data(args[1])

            if len(x_series) != 1:
                raise ValueError("There should be exactly one series for x-axis")
            if not series:
                raise ValueError("No series found")
            x_series = x_series[0]
            if x_name is None:
                x_name = x_name_[0]
            if names is None:
                names = names_
        elif isinstance(args[0], list):
            series = args[0]
            x_series = args[1]

    if series is None:
        raise ValueError("scatter should be called with an indicator collection"
                         " or a series. Check documentation for details.")

    if x_name is None:
        x_name = 'x'
    if names is None:
        names = [f'{i + 1}' for i in range(len(series))]

    tables = [_binned_heatmap.data_to_table(s, x_series) for s in series]
    names = _remove_names_prefix(names)

    return _binned_heatmap.render(
        tables,
        names=names,
        x_name=x_name,
        width=width,
        height=height,
        height_minimap=height_minimap)


def indicator_data(indicators: IndicatorCollection) -> Tuple[List[np.ndarray], List[List[str]]]:
    r"""
    Returns a tuple of a list of series and a list of names of series.
    Each series, `S` is a timeseries of histograms of shape `[T, 10]`,
    where `T` is the number of timesteps.
    `S[:, 0]` is the `global_step`.
    `S[:, 1:10]` represents the distribution at basis points
    `0, 6.68, 15.87, 30.85, 50.00, 69.15, 84.13, 93.32, 100.00`.

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.indicator_data(indicators)
    """

    series, names = _cache.get_indicators_data(indicators)

    if not series:
        raise ValueError("No series found")

    return series, names


def artifact_data(indicators: IndicatorCollection) -> Tuple[List[any], List[str]]:
    r"""
    Returns a tuple of a list of series and a list of names of series.
    Each series, ``S`` is a timeseries of histograms of shape ``[T, 10]``,
    where ``T`` is the number of timesteps.
    ``S[:, 0]`` is the `global_step`.
    ``S[:, 1:10]`` represents the distribution at basis points:
    ``0, 6.68, 15.87, 30.85, 50.00, 69.15, 84.13, 93.32, 100.00``.

    Example:
        >>> from labml import analytics
        >>> indicators = analytics.runs('1d3f855874d811eabb9359457a24edc8')
        >>> analytics.artifact_data(indicators)
    """

    series, names = _cache.get_artifacts_data(indicators)

    if not series:
        raise ValueError("No series found")

    return series, names
