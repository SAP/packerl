import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from tqdm import tqdm


def deep_update(source, overrides):
    """
    Deep-update a nested dictionary or similar mapping.
    """
    for new_k, new_v in overrides.items():
        if not isinstance(source, Mapping):  # leaf case
            source = overrides
        elif isinstance(new_v, Mapping):
            deep_updated = deep_update(source.get(new_k, {}), new_v)
            source[new_k] = deep_updated
        else:
            source[new_k] = overrides[new_k]
    return source


def flatten_nested_list(nested_list):
    """
    Flatten a nested list.
    """
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_nested_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def list_of_dict_to_dict_of_lists(list_of_dict):
    """
    Convert a list of dictionaries to a dictionary of lists
    """
    dict_of_lists = {}
    for key in list_of_dict[0].keys():
        dict_of_lists[key] = [d[key] for d in list_of_dict]
    return dict_of_lists


def dict_of_lists_to_dict_of_stats(dict_of_list):
    """
    Convert a dictionary of run statistics probivided via value lists,
     to a dictionary of statistics by calculating mean, median, min, max and std from each value list.
    """
    dict_of_stats = {}
    for metric_name, metric_values in dict_of_list.items():
        dict_of_stats[f'{metric_name}_mean'] = np.mean(metric_values)
        dict_of_stats[f'{metric_name}_median'] = np.median(metric_values)
        dict_of_stats[f'{metric_name}_min'] = np.min(metric_values)
        dict_of_stats[f'{metric_name}_max'] = np.max(metric_values)
        dict_of_stats[f'{metric_name}_std'] = np.std(metric_values)
    return dict_of_stats


def pprint_dict(d, indent: int = 0):
    """
    Pretty-print a dictionary.
    """
    for key, value in d.items():
        indent_str = ' ' * indent
        if isinstance(value, dict):
            print(f"{indent_str}{str(key)}: ")
            pprint_dict(value, indent + 4)
        else:
            print(f"{indent_str}{str(key)}: {str(value)}")


def pop_or_none(l):
    """
    return the last element of a list, or None if the list is empty
    """
    return l.pop() if len(l) > 0 else None


def ensure_sysctl_value(sysctl_var_name, desired_value):
    """
    Ensure that a sysctl variable has a certain value, if not, try to set it via sudo.
    :param sysctl_var_name: name of the sysctl variable
    :desired_value: desired value of the sysctl variable
    """
    process = subprocess.Popen(['sysctl', sysctl_var_name], stdout=subprocess.PIPE)
    sysctl_var_value = int(process.communicate()[0].decode('utf-8').split('=')[-1].strip())
    if sysctl_var_value > desired_value:
        print(f"Missing kernel permissions ({sysctl_var_name}={sysctl_var_value}) "
              f"-> trying to temporally set via sudo...")
        try:
            _ = subprocess.Popen(['sudo', 'sysctl', f'{sysctl_var_name}={str(desired_value)}'],
                                 stdout=subprocess.PIPE)
        except Exception as e:
            print(f"Couldn't set variable, showing error stacktrace and aborting... \n{e}")
            exit(1)


def listify(val) -> list:
    """
    Returns a list containing val if it's not already a list.
    """
    return val if isinstance(val, list) else [val]


def pretty_repr(d: dict, indent: int = 0) -> List[str]:
    """
    Returns a human-readable representation of a (nested) dict in the form of string lists.
    :param d: The dict.
    :param indent: Each sublevel of the dict is indented by this amount of characters.
    :return: A human-readable presentation of a nested dict in the form of string lists.
    """
    lines = []
    for key, val in d.items():
        if isinstance(val, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.extend(pretty_repr(val, indent + 3))
        else:
            lines.append(f"{' ' * indent}{key}: {val}")
    return lines


def pretty_print(d: dict):
    """
    Prints a human-readable representation of a given (nested) dict.
    :param d: The dict.
    """
    d_repr = '\n'.join(pretty_repr(d))
    print(f"\n{d_repr}\n")


def merge_dicts(d1, d2):
    """
    merges a possibly nested dict d2 into d1.
    """
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            merge_dicts(d1[key], value)
        else:
            d1[key] = value


def write_to_disk(G: nx.Graph, tms_file_lines: List[str], graph_dir: Path,
                  print_to_console: bool = False):
    """
    Write a given network (graph, netconfig, traffic) to the disk.
    :return:
    """
    if print_to_console:
        print(f"writing graph data to {graph_dir}")

    # save graph information
    nx.write_graphml_xml(G, str((graph_dir / "graph_attr.graphml").resolve()))

    # save traffic matrices
    graph_tm_dir = graph_dir / "TM"
    graph_tm_dir.mkdir()
    for i, tm_file_lines in enumerate(tms_file_lines):
        with open(graph_tm_dir / f"TM-{i}", "w+") as tm_file:
            tm_file.writelines([line + "\n" for line in tm_file_lines])


class TqdmUpTo(tqdm):
    r"""
    A wrapper class around the tqdm progress bar that can be used for showing download progress.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Taken from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5,
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [mentioned here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        r"""
        Updates the tqdm progress indicator.
        b (int): Number of blocks transferred so far [default: 1].
        bsize (int): Size of each block (in tqdm units) [default: 1].
        tsize (int): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_from_url(url: str, dst_path: str):
    r"""
    Downloads the contents of specified URL to the specified destination filepath.
    Uses :class:`TqdmUpTo` to show download progress.
    Args:
        url (str): The URL to download from.
        dst_path (str): The path to save the downloaded data to.
    """
    from urllib.request import urlretrieve
    print(f"Downloading from {url}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]) as t:
        urlretrieve(url, dst_path, reporthook=t.update_to)


def rescale(values: np.ndarray, to_min, to_max):
    """
    Rescales the values of a given array from its original value range to a new, specified range.
    value: The array to rescale.
    to_min: The minimum value of the new range.
    to_max: The maximum value of the new range.
    """

    from_min = np.min(values)
    from_max = np.max(values)

    # If the original range is a single value, return the new value as the middle of the new range
    if from_min == from_max:
        to_med = (to_min + to_max) / 2
        return np.array([to_med] * len(values))

    # Calculate the proportion of the value within the original range
    proportion = (values - from_min) / (from_max - from_min)

    # Map the proportion to the new range and return the result
    scaled_value = proportion * (to_max - to_min) + to_min
    return scaled_value
