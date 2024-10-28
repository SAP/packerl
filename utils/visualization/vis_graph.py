import os
import tempfile
from copy import deepcopy
from io import BytesIO
import subprocess
import math

import pydot
from PIL import Image, ImageDraw

from features.features import ALL_EDGE_FEATURES
from utils.visualization.vis_utils import (get_node_color, hex_to_rgb_tuple,
                                           feat_to_alpha_value, angle_to_x_axis)


def queue_load_to_color(loads_and_colors):
    """
    :return: A small DOT/HTML color representation of the queue load. It is used for coloring the outgoing edge.
    """
    if all(load < 0.001 for load, _ in loads_and_colors):
        return "gray60"
    color_strs = []
    max_load = 0.0
    for load, color in loads_and_colors:
        if load > max_load:
            color_strs.append(f"{color};{load - max_load}")
            max_load = load
    color_strs.append("gray60")
    return ":".join(color_strs)


def channel_config_to_str(datarate, delay, timestep_duration_in_s, packet_size_with_headers):
    """
    :return: A small HTML string representation for use in the DOT graph format
    that contains the link's datarate and delay.
    """
    if delay >= 1:
        dl_unit = 'ms'
        dl = delay
    else:
        dl_unit = 'µs'
        dl = delay * 1000
    dr = int(datarate * timestep_duration_in_s // (packet_size_with_headers * 8))
    return f"{dr}p<br/>{dl}{dl_unit}"  # read: 'up to <dr> packets sendable this timestep at delay of <dl><dl_unit>'


def create_dot_vis(graph_name, directed, title_label, nodes, edges, render_cfg) -> Image.Image:
    """
    Creates a DOT graph using the given information, renders it and returns a numpy RGB array.
    :return: A numpy array containing the rendered graph as an RGB-PNG image
    """

    fontsize_multiplier = render_cfg.get("fontsize_multiplier", 1)

    graph = pydot.Dot(graph_name,
                      graph_type="digraph" if directed else "graph",
                      label=title_label,
                      labelloc="t",
                      fontname="Helvetica-Bold",
                      layout="neato")

    graph.set_node_defaults(shape="circle",
                            style="filled",
                            fontname="Courier",
                            penwidth=fontsize_multiplier * render_cfg['penwidth'])

    graph.set_edge_defaults(fontname="Courier",
                            penwidth=fontsize_multiplier * render_cfg['penwidth'],
                            arrowsize=fontsize_multiplier * render_cfg['edge_arrow_size'],
                            color="gray60",
                            labeldistance=fontsize_multiplier * render_cfg['edge_label_dist'],
                            labelangle=render_cfg['edge_label_angle'])

    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge)

    p = subprocess.run(['dot', "-Tpng"],
                       stdout=subprocess.PIPE,
                       input=graph.to_string().encode('utf-8')
                       )
    return Image.open(BytesIO(p.stdout)).convert("RGB")


def visualize_monitoring_nx(monitoring_nx, render_node_pos, render_cfg, ep_name, t,
                            packet_size_with_headers, use_flow_control):
    """
    Given a monitoring graph, assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given nx monitoring graph as an RGB-PNG image.
    """
    render_cfg["fontsize_multiplier"] = 1 / math.log(monitoring_nx.number_of_nodes(), 5)

    # title
    timestep_duration_in_s = round(monitoring_nx.graph['elapsedTime'], 3)
    title_label = f"monitoring: {ep_name}, t={t}, duration={timestep_duration_in_s}s, \n" \
                  f"avg. delay = {round(monitoring_nx.graph['avgPacketDelay'], 3)}, " \
                  f"T: {monitoring_nx.graph['sentPackets']}, " \
                  f"R: {monitoring_nx.graph['receivedPackets']}, " \
                  f"D: {monitoring_nx.graph['droppedPackets']}"

    # node data
    node_spacing = render_cfg['node_spacing']
    nodes = []
    node_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['node_fontsize']
    for node_id, node_attrs in list(monitoring_nx.nodes(data=True)):
        node_label = f"<<FONT POINT-SIZE=\"{node_fontsize * 2}\"><b>{node_id}</b></FONT><br/>" \
                     f"<FONT POINT-SIZE=\"{node_fontsize}\">T:{node_attrs['sentPackets']:5d}<br/>" \
                     f"R:{node_attrs['receivedPackets']:5d}</FONT>>"
        node_pos = render_node_pos[node_id]
        nodes.append(pydot.Node(str(node_id), label=node_label, fillcolor='white',
                          pos=f"{node_spacing * node_pos[0]},{node_spacing * node_pos[1]}!"))

    # edge data
    edges = []
    edge_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['edge_fontsize']
    for u, v, edge_attrs in list(monitoring_nx.edges(data=True)):
        # edge color
        if use_flow_control:
            queue_load_color = queue_load_to_color([(edge_attrs['txQueueLastLoad'], "OrangeRed1"),
                                                    (edge_attrs['txQueueMaxLoad'], "brown"),
                                                    (edge_attrs['queueDiscLastLoad'], "orange"),
                                                    (edge_attrs['queueDiscMaxLoad'], "PeachPuff")])
        else:
            queue_load_color = queue_load_to_color([(edge_attrs['txQueueLastLoad'], "OrangeRed1"),
                                                    (edge_attrs['txQueueMaxLoad'], "brown")])

        # edge taillabel
        taillabel = f"<<FONT POINT-SIZE=\"{edge_fontsize}\">" \
                    f"T:{edge_attrs['sentPackets']:5d}<br/>" \
                    f"R:{edge_attrs['receivedPackets']:5d}<br/>" \
                    f"D:{edge_attrs['droppedPackets']:5d}</FONT>>"

        # edge label
        channel_config_str = channel_config_to_str(edge_attrs['channelDataRate'], edge_attrs['channelDelay'],
                                                   timestep_duration_in_s, packet_size_with_headers)
        label = f"<<FONT POINT-SIZE=\"{edge_fontsize}\" COLOR=\"blue\"><b>{channel_config_str}</b></FONT>>"
        edges.append(pydot.Edge(str(u), str(v), label=label, taillabel=taillabel, color=queue_load_color))

    return create_dot_vis("monitoring", True, title_label, nodes, edges, render_cfg)


def visualize_state_pyg(render_cfg, graph, render_node_pos, acceptable_features, name):
    """
    Given a Pytorch geometric state graph, assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given Pytorch geometric
    state graph as an RGB-PNG image.
    """

    render_cfg["fontsize_multiplier"] = 1 / math.log(graph.num_nodes, 5)

    # title with global features
    title_label_lines = [name]
    global_feats = [round(global_feat, 4) for global_feat in graph.u.flatten().tolist()]
    if len(global_feats) > 0:
        global_feat_strs = [f'{gf_name}: {gf}' for (gf_name, gf) in zip(acceptable_features['global'], global_feats)]
        title_label_lines.append(', '.join(global_feat_strs))
    if len(acceptable_features['node']) > 0:
        title_label_lines.append(f"node feat: {acceptable_features['node']}")
    if len(acceptable_features['edge']) > 0:
        title_label_lines.append(f"edge feat: {acceptable_features['edge']}")
    title_label = "\n".join(title_label_lines)

    # nodes with features
    node_spacing = render_cfg['node_spacing']
    nodes = []
    node_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['node_fontsize']
    for node_id in range(graph.num_nodes):
        node_feat_strs = ([f"<FONT POINT-SIZE=\"{node_fontsize * 2}\"><b>{node_id}</b></FONT>"]
                          + [f"<FONT POINT-SIZE=\"{node_fontsize}\">{str(round(feat, 3))}</FONT>"
                             for feat in graph.x[node_id].tolist()])
        node_label = f"<{'<br/>'.join(node_feat_strs)}>"
        node_pos = render_node_pos[node_id]
        nodes.append(pydot.Node(str(node_id), label=node_label, fillcolor='white',
                                pos=f"{node_spacing * node_pos[0]},{node_spacing * node_pos[1]}!"))

    # edges with features
    edges = []
    edge_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['edge_fontsize']
    source_idx, dest_idx = graph.edge_index.tolist()
    for edge_id, (u, v) in enumerate(zip(source_idx, dest_idx)):
        edge_feats = [round(feat, 3) for feat in graph.edge_attr[edge_id].tolist()]
        if len(edge_feats) > 0:
            edge_feat_strs = []
            for ef_name, ef in zip(acceptable_features["edge"], edge_feats):
                if ef < 0.5 or ALL_EDGE_FEATURES[ef_name][0] != "unit":  # only color 'unit' range feat > 0.5
                    edge_feat_str = f'<FONT COLOR=\"black\"><b>{ef}</b></FONT>'
                elif ef >= 1.0:
                    edge_feat_str = f'<FONT COLOR=\"OrangeRed1\"><b>{ef}</b></FONT>'
                else:
                    edge_feat_str = f'<FONT COLOR=\"firebrick4\">{ef}</FONT>'
                edge_feat_strs.append(edge_feat_str)
            taillabel = f"<<FONT POINT-SIZE=\"{edge_fontsize}\">{'<br/>'.join(edge_feat_strs)}</FONT>>"
            edges.append(pydot.Edge(str(u), str(v), taillabel=taillabel))
        else:
            edges.append(pydot.Edge(str(u), str(v)))

    # create graph and visualize
    new_config = deepcopy(render_cfg)
    new_config['edge_label_dist'] = 6
    new_config['edge_label_angle'] = 18
    return create_dot_vis("state_pyg", True, title_label, nodes, edges, new_config)


def visualize_action_raw_pyg(render_cfg, graph, action, render_node_pos, name):
    """
    Given a Pytorch geometric unnormalized action graph,
    assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given Pytorch geometric
    unnormalized action graph as an RGB-PNG image.
    """

    # we visualize differently, depending on whether the action is next-hop-centric or link-weight-centric
    is_next_hop_action = action.ndim == 2
    render_cfg["fontsize_multiplier"] = 1 / math.log(graph.num_nodes, 5)
    if is_next_hop_action:
        actions_raw = action.clone()[:, :graph.num_nodes]
    else:
        actions_raw = action

    # nodes with id and color
    node_spacing = render_cfg['node_spacing']
    nodes = []
    node_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['node_fontsize']
    for node_id in range(graph.num_nodes):
        node_label = f"<<FONT POINT-SIZE=\"{node_fontsize * 2}\" ><b>{node_id}</b></FONT>>"
        node_pos = render_node_pos[node_id]
        nodes.append(pydot.Node(str(node_id), label=node_label, fillcolor=get_node_color(node_id),
                                pos=f"{node_spacing * node_pos[0]},{node_spacing * node_pos[1]}!"))

    # edges with output values, matching node colors
    edges = []
    edge_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['edge_fontsize']
    source_idx, dest_idx = graph.edge_index.tolist()
    if is_next_hop_action:  # visualize values per next-hop per destination node
        for edge_id, (u, v) in enumerate(zip(source_idx, dest_idx)):
            edge_feat = actions_raw[edge_id].tolist()
            taillabel_lines = [f'<FONT COLOR=\"{get_node_color(i)}\"><b>{i}</b>: {round(feat, 3)}</FONT>'
                                    for i, feat in enumerate(edge_feat) if i != u]
            taillabel = f"<<FONT POINT-SIZE=\"{edge_fontsize}\">{'<br/>'.join(taillabel_lines)}</FONT>>"
            edges.append(pydot.Edge(str(u), str(v), taillabel=taillabel))
    else:  # visualize link weights
        for edge_id, (u, v) in enumerate(zip(source_idx, dest_idx)):
            weight_raw = actions_raw[edge_id].item()
            weight_actual = math.exp(min(weight_raw, 10))
            taillabel = (f"<<FONT POINT-SIZE=\"{edge_fontsize}\"><b>{round(weight_actual, 2)}</b><br/>"
                         f"raw: {round(weight_raw, 3)}</FONT>>")
            edges.append(pydot.Edge(str(u), str(v), taillabel=taillabel))

    # create graph and visualize
    new_cfg = deepcopy(render_cfg)
    new_cfg['edge_label_dist'] = 6
    new_cfg['edge_label_angle'] = 18
    return create_dot_vis(name, True, name, nodes, edges, new_cfg)


def visualize_action_nrm_pyg(render_cfg, graph, action, render_node_pos, name):
    """
    Given a Pytorch geometric normalized action graph,
    assemble the information needed to create a DOT graph and render it.
    :return: A numpy array containing a visualization of the given Pytorch geometric
    normalized action graph as an RGB-PNG image.
    """

    render_cfg["fontsize_multiplier"] = 1 / math.log(graph.num_nodes, 5)

    # nodes with id and color
    node_spacing = render_cfg['node_spacing']
    nodes = []
    node_fontsize = render_cfg["fontsize_multiplier"] * render_cfg['node_fontsize']
    for node_id in range(graph.num_nodes):
        node_label = f"<<FONT POINT-SIZE=\"{node_fontsize * 2}\"><b>{node_id}</b></FONT>>"
        node_pos = render_node_pos[node_id]
        nodes.append(pydot.Node(str(node_id), label=node_label, fillcolor=get_node_color(node_id),
                                pos=f"{node_spacing * node_pos[0]},{node_spacing * node_pos[1]}!"))

    arrow_H = max(2, round(render_cfg["fontsize_multiplier"] * render_cfg['edge_fontsize'] * 2))
    arrow_W = max(2, round(render_cfg["fontsize_multiplier"] ** 2 * render_cfg['edge_fontsize'] * 2))

    def dest_to_img(colors_and_feat, angle):
        """
        auxiliary function to create an image containing the routing arrow indicators
        """
        # Create a new image with the exact required size
        image = Image.new('RGBA', (2 * arrow_W * len(colors_and_feat), arrow_H), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        for i, (color, feat) in enumerate(colors_and_feat):
            (r, g, b), a = hex_to_rgb_tuple(color), feat_to_alpha_value(feat)
            draw.polygon([(i*arrow_W, 0), (i*arrow_W, arrow_H), ((i+1)*arrow_W, arrow_H // 2)], fill=(r, g, b, a))

        # rotate
        image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255, 0))
        return image

    def get_edge_label_img(src_idx, edge_feat, angle, img_filename):
        """
        auxiliary function to create edge label images containing the routing arrow indicators
        """
        edge_dest = [i for i in range(len(edge_feat)) if i != src_idx]
        if len(edge_dest) > 0:
            edge_dest_img = dest_to_img([(get_node_color(i), edge_feat[i]) for i in edge_dest], angle)
            edge_dest_img.save(img_filename)
            return f"<<TABLE border='0'><TR><TD><IMG SRC=\"{img_filename}\"/></TD></TR></TABLE>>", img_filename
        else:
            return None, None

    # undirected edges with arrows indicating packet routing decisions per edge
    edges = []
    edge_idx_dict = dict()
    for i, edge_idx in enumerate(list(zip(*graph.edge_index.tolist()))):
        edge_idx_dict[edge_idx] = i
    with tempfile.TemporaryDirectory() as td:
        while len(edge_idx_dict) > 0:
            (u, v), edge_id = edge_idx_dict.popitem()
            edge_id_rev = edge_idx_dict.pop((v, u))

            edge_feat = action[edge_id].tolist()
            edge_feat_rev = action[edge_id_rev].tolist()

            angle = angle_to_x_axis((render_node_pos[v][1] - render_node_pos[u][1]),
                                    (render_node_pos[v][0] - render_node_pos[u][0]))
            angle_rev = angle - 180

            edge_labels = dict()
            edge_tl, tl_img_fp = get_edge_label_img(u, edge_feat, angle, os.path.join(td, f"e_{u}_{v}.png"))
            if tl_img_fp is not None:
                edge_labels["taillabel"] = edge_tl
            edge_hl, hl_img_fp = get_edge_label_img(v, edge_feat_rev, angle_rev, os.path.join(td, f"e_{v}_{u}.png"))
            if hl_img_fp is not None:
                edge_labels["headlabel"] = edge_hl
            edge = pydot.Edge(str(u), str(v), **edge_labels)
            edges.append(edge)

        # create graph and visualize
        new_cfg = deepcopy(render_cfg)
        new_cfg['edge_label_dist'] = 8
        new_cfg['edge_label_angle'] = 0
        dot_vis = create_dot_vis(name, False, name, nodes, edges, new_cfg)
    return dot_vis
