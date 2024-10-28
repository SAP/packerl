"""
Non-configurable constants
"""

"""
Base directory for ns3
"""
NS3_DIR = "./ns3/"

"""
Base output directory for PackeRL
"""
DEFAULT_OUT_DIR = "../packerl_out"

"""
Relative filepath of default experiment config file
"""
DEFAULT_CONFIG_FP = "config/defaults.yaml"

"""
Relative filepath of default save location for run config
"""
RUN_CONFIG_FILENAME = "run_config.yaml"

"""
Size of the IPv4 header in bytes
"""
IPV4_HEADER_SIZE = 20

"""
Size of the ICMP header in bytes
"""
ICMP_HEADER_SIZE = 8

"""
NS3-AI shared memory size in bytes
"""
SHM_SIZE = 2 ** 24