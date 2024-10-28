from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import yaml
from typing import List, Union


def serialize_tm(tm, as_int=False):
    """
    Serialize a traffic matric into a multi-line string, where each line contains a demand.
    """
    N = tm.shape[0]
    demands = []

    for i in range(N):
        for j in range(N):
            if i != j:  # Skip diagonal elements
                demand_value = round(tm[i, j]) if as_int else tm[i, j]
                demands.append(f'demand_{len(demands)} {i} {j} {demand_value}')

    return f'DEMANDS {len(demands)}\nlabel src dest bw\n' + '\n'.join(demands)


def serialize_topology_txt(graph, capacity_scaling=1.0, as_int=False, directed=False):
    """
    Serialize a network topology into a multi-line string, where each line contains an edge.
    """
    edge_lines = []
    for i, (u, v, attrs) in enumerate(list(graph.edges(data=True))):
        datarate = capacity_scaling * attrs['channelDataRate']
        dr = max(round(datarate), 1) if as_int else datarate
        dl = max(round(attrs['channelDelay']), 1) if as_int else attrs['channelDelay']
        link_i = 2*i if directed else i
        edge_lines.append(f"Link_{link_i} {u} {v} 1 {dr} {dl}")
        if directed:
            edge_lines.append(f"Link_{link_i+1} {v} {u} 1 {dr} {dl}")
    return f"EDGES {len(graph.edges)}\nlabel src dest weight bw delay\n" + '\n'.join(edge_lines)


def serialize_topology_graphml(graph):
    """
    Serialize a network topology to the graphml format.
    """
    # Create the root element of the XML
    root = Element('graphml')
    root.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:schemaLocation', 'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd')

    # Add key elements for edge attributes (delay and datarate)
    _ = SubElement(root, 'key', id="d0", for_="edge", attr_name="delay", attr_type="double")
    _ = SubElement(root, 'key', id="d1", for_="edge", attr_name="datarate", attr_type="double")

    # Add the graph element
    graph_element = SubElement(root, 'graph', edgedefault="undirected")

    # Add nodes to the graph
    for node in graph.nodes():
        _ = SubElement(graph_element, 'node', id=str(node))

    # Add edges with attributes to the graph
    for src, dst, attrs in graph.edges(data=True):
        edge_element = SubElement(graph_element, 'edge', source=str(src), target=str(dst))
        delay_data = SubElement(edge_element, 'data', key="d0")
        delay_data.text = str(attrs['channelDelay'])
        datarate_data = SubElement(edge_element, 'data', key="d1")
        datarate_data.text = str(attrs['channelDataRate'])

    # Generate a prettified XML string
    xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    return xml_str


class QuotedString(str):
    """
    A custom string class that forces quotes around the string representation when dumping to YAML.
    """
    pass


def convert_strings_to_quoted_strings(obj):
    """
    Recursively converts all strings in a dictionary to QuotedString objects.
    """
    if isinstance(obj, dict):
        return {k: convert_strings_to_quoted_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_strings_to_quoted_strings(element) for element in obj]
    elif isinstance(obj, str):
        return QuotedString(obj)
    else:
        return obj


def _yaml_universal_quoted_str_representer(dumper, data):
    """
    Force quotes around string representations.
    """
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')


def yaml_dump_quoted(data: Union[dict, List[dict]], yaml_file, default_flow_style=False, sort_keys=True):
    """
    Dumps one or more dictionaries to an already opened YAML file, ensuring all strings are quoted.
    """

    # Register the custom representer for this specific dump operation and mark convert leaf strings to QuotedString
    yaml.add_representer(QuotedString, _yaml_universal_quoted_str_representer)
    data = convert_strings_to_quoted_strings(data)

    if isinstance(data, dict):

        yaml.dump(data, yaml_file, default_flow_style=default_flow_style, sort_keys=sort_keys)
    else:
        yaml.dump_all(data, yaml_file, default_flow_style=default_flow_style, sort_keys=sort_keys)
