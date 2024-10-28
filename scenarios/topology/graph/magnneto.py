from pathlib import Path
import zipfile
import re
import shutil
from typing import List

import networkx as nx
import numpy as np

from utils.utils import download_from_url
from scenarios.config import BaseConfig


DATA_URL = "https://bnn.upc.edu/download/magnneto-te_datasets/"
DATA_FOLDER_NAME = 'magnneto'
ARCHIVE_PATH_FOLDER_NAME = 'magnneto-te_datasets'
ARCHIVE_NAME = 'magnneto-te_datasets.zip'
AVAILABLE_NETWORKS = ['GBN', 'NSFNet', 'GEANT2']


class MagnnetoDatasetConfig(BaseConfig):
    """
    Configuration for Magnneto dataset graph generation.
    """
    data_dir: str
    networks: List[str]


def check_for_magnneto_data(data_path):
    """
    Checks whether the magnneto dataset is available in the given data path. If not, downloads and extracts it.
    """

    # if all graph_attr.txt files are already available we can return
    if all([(data_path / DATA_FOLDER_NAME / network / "graph_attr.txt").exists() for network in AVAILABLE_NETWORKS]):
        return

    # download and extract the entire dataset to a separate folder, so that we can remove unnecessary files later
    magnneto_archive_folder_path = data_path / ARCHIVE_PATH_FOLDER_NAME
    if not magnneto_archive_folder_path.exists():
        magnneto_archive_folder_path.mkdir()
    magnneto_archive_path = magnneto_archive_folder_path / ARCHIVE_NAME
    if not magnneto_archive_path.exists():
        download_from_url(DATA_URL, str(magnneto_archive_path.resolve()))
    if magnneto_archive_path in magnneto_archive_folder_path.glob('*'):
        with zipfile.ZipFile(magnneto_archive_path, 'r') as zip_file:
            zip_file.extractall(magnneto_archive_folder_path)

    # move the required graph_attr.txt files to the actual magnneto_data_path (create that if not yet there)
    magnneto_data_path = data_path / DATA_FOLDER_NAME
    if not magnneto_data_path.exists():
        magnneto_data_path.mkdir()
    for network in AVAILABLE_NETWORKS:
        datasets_path = magnneto_archive_folder_path / "datasets" / network
        for file in datasets_path.glob('*'):
            if re.match(r'graph_attr.txt', file.name):
                new_network_path = magnneto_data_path / network
                if not new_network_path.exists():
                    new_network_path.mkdir()
                file.rename(new_network_path / file.name)

    # remove the rest
    shutil.rmtree(magnneto_archive_folder_path)


def load_magnneto_topology(graph_filename, chosen_network):
    """
    Loads a graph from the Magnneto dataset.
    """
    G_raw = nx.read_gml(graph_filename, destringizer=int)
    G = nx.from_scipy_sparse_array(nx.adjacency_matrix(G_raw))
    G.graph['name'] = chosen_network

    # link datarates range from
    link_datarates = {(u, v): int(attrs["bandwidth"]) for u, v, attrs in G_raw.edges(data=True)}
    nx.set_edge_attributes(G, link_datarates, 'datarate')
    G.graph['keepDatarateProportions'] = True  # consumed by the attribute generator to rescale the datarates properly

    return G


def generate_magnneto(data_dir: str, networks: List[str], rng: np.random.Generator, **kwargs):
    """
    Generates a random graph by choosing one of [NSFNet, GEANT2] and pre-processing it.
    """
    # check whether the magnneto data is available, and download it if not
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError("data_dir does not exist - perhaps you forgot to set it?")
    check_for_magnneto_data(data_path)

    # choose a graph from the graph zoo dataset
    chosen_network = rng.choice(networks)
    graph_filename = str((data_path / DATA_FOLDER_NAME / chosen_network / "graph_attr.txt").resolve())

    # load the chosen graph and return it
    return load_magnneto_topology(graph_filename, chosen_network)
