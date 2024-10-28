import io
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter
from PIL import Image

from scenarios.events.traffic.traffic_demand import TrafficDemand

from utils.visualization.vis_graph import visualize_state_pyg, visualize_action_raw_pyg, visualize_action_nrm_pyg
from utils.visualization.vis_utils import pad_images_to_largest, grid_images


def visualize_step(render_config, acceptable_features, node_rendering_positions,
                   rendered_monitoring, state_pyg, action_raw, action_discretized):
    """
    Visualize a single environment step, including the rendered monitoring image,
    as well as visualizations for the NN input and action output.
    """
    if render_config.get('full_step_vis', False):
        training_vis_imgs = [
            rendered_monitoring,
            visualize_state_pyg(render_config, state_pyg, node_rendering_positions,
                                acceptable_features, "state_pyg"),
            visualize_action_raw_pyg(render_config, state_pyg, action_raw, node_rendering_positions,
                                 "action_pyg"),
            visualize_action_nrm_pyg(render_config, state_pyg, action_discretized, node_rendering_positions,
                                     "action_pyg_normalized")
        ]
    else:
        training_vis_imgs = [
            rendered_monitoring,
            visualize_action_nrm_pyg(render_config, state_pyg, action_discretized, node_rendering_positions,
                                     "action_pyg_normalized")
            ]
    padded_imgs = pad_images_to_largest(training_vis_imgs)
    return grid_images(padded_imgs)


def visualize_traffic_events(events) -> Image.Image:
    """
    Visualize the traffic events in the given list of events.
    """
    demand_ts_and_amounts = [(e.t, e.amount)
                             for events_t in events
                             for e in events_t
                             if isinstance(e, TrafficDemand)
                             ]
    demand_ts, demand_amounts = zip(*demand_ts_and_amounts)

    # Create the scatter plot
    plt.figure(figsize=(16, 9))
    plt.scatter(demand_ts, demand_amounts)

    # Set axis labels
    plt.xlabel('Arrival time (ms)')
    plt.ylabel('Amount (bytes)')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(LogFormatter())  # Improve log scale formatting if needed

    # write the plot to a buffer and create an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    traffic_vis: Image.Image = Image.open(buf).copy().convert("RGB")  # copy so that the buffer can be closed

    # cleanup
    plt.close()
    buf.close()
    return traffic_vis


def visualize_traffic_matrices(observed_traffic: List[np.ndarray]):
    """
    Visualize the given traffic matrices as a series of heatmaps.
    """

    observed_traffic_stacked = np.array(observed_traffic)
    max_traffic = np.max(observed_traffic_stacked)

    visualized_frames = []

    for t, observed_traffic_t in enumerate(observed_traffic):
        fig, ax = plt.subplots()
        cax = ax.matshow(observed_traffic_t, vmin=0, vmax=max_traffic)
        fig.colorbar(cax)
        ax.set_title(f"sent bytes ({t=})")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        traffic_vis: Image.Image = Image.open(buf).copy().convert("RGB")   # copy so that the buffer can be closed
        visualized_frames.append(traffic_vis)

        plt.close(fig)
        buf.close()

    return visualized_frames


def visualize_ep_performance(ep_name, ep_monitoring_count, ep_stats, plot_annotation_fontsize):
    """
    Creates network performance plots concerning e.g. avg. packet delay, link utilization, packet stats.
    :param plot_annotation_fontsize: Fontsize for plot curve annotations
    :return: A dictionary containing the RGB arrays of the created plots (keyed by their names)
    """

    ep_plot_dict = dict()
    plot_x = list(range(1, ep_monitoring_count + 1))

    # sent/received/dropped/retransmitted packets: 'packet_stats'
    fig, ax = plt.subplots()
    ax.set_title(f"{ep_name}: sent/received/dropped/retransmitted packets")
    ax.set_xlim(1, ep_monitoring_count)
    ax.set_xlabel('t')
    ax.plot(plot_x, ep_stats['sentPackets'], label="sent packets", color='blue')
    sent_packets_sum = ep_stats['sentPackets'][-1]
    ax.annotate(str(sent_packets_sum), xy=(plot_x[-1], sent_packets_sum), color='blue',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['receivedPackets'], label="received packets", color='green')
    received_packets_sum = ep_stats['receivedPackets'][-1]
    ax.annotate(str(received_packets_sum), xy=(plot_x[-1], received_packets_sum), color='green',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['droppedPackets'], label="dropped packets", color='orange')
    dropped_packets_sum = ep_stats['droppedPackets'][-1]
    ax.annotate(str(dropped_packets_sum), xy=(plot_x[-1], dropped_packets_sum), color='orange',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['retransmittedPackets'], label="retransmitted packets", color='purple')
    retransmitted_packets_sum = ep_stats['retransmittedPackets'][-1]
    ax.annotate(str(retransmitted_packets_sum), xy=(plot_x[-1], retransmitted_packets_sum), color='purple',
                fontsize=plot_annotation_fontsize)
    ax.legend()
    fig.canvas.draw()
    ep_plot_dict['packet_stats'] = Image.frombytes('RGB',
                                                   fig.canvas.get_width_height(),
                                                   fig.canvas.tostring_rgb()
                                                   )
    plt.close()

    # sent/received/dropped/retransmitted bytes: 'byte_stats'
    fig, ax = plt.subplots()
    ax.set_title(f"{ep_name}: sent/received/dropped/retransmitted bytes")
    ax.set_xlim(1, ep_monitoring_count)
    ax.set_xlabel('t')
    ax.plot(plot_x, ep_stats['sentBytes'], label="sent bytes", color='blue')
    sent_bytes_sum = ep_stats['sentBytes'][-1]
    ax.annotate(str(sent_bytes_sum), xy=(plot_x[-1], sent_bytes_sum), color='blue',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['receivedBytes'], label="received bytes", color='green')
    received_bytes_sum = ep_stats['receivedBytes'][-1]
    ax.annotate(str(received_bytes_sum), xy=(plot_x[-1], received_bytes_sum), color='green',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['droppedBytes'], label="dropped bytes", color='orange')
    dropped_bytes_sum = ep_stats['droppedBytes'][-1]
    ax.annotate(str(dropped_bytes_sum), xy=(plot_x[-1], dropped_bytes_sum), color='orange',
                fontsize=plot_annotation_fontsize)
    ax.plot(plot_x, ep_stats['retransmittedBytes'], label="retransmitted bytes", color='purple')
    retransmitted_bytes_sum = ep_stats['retransmittedBytes'][-1]
    ax.annotate(str(retransmitted_bytes_sum), xy=(plot_x[-1], retransmitted_bytes_sum), color='purple',
                fontsize=plot_annotation_fontsize)
    ax.legend()
    fig.canvas.draw()
    ep_plot_dict['byte_stats'] = Image.frombytes('RGB',
                                                 fig.canvas.get_width_height(),
                                                 fig.canvas.tostring_rgb()
                                                 )
    plt.close()

    # avg. delay: 'avg_delay'
    fig, ax = plt.subplots()
    ax.set_title(f"{ep_name}: avg. total packet delay")
    ax.set_xlim(1, ep_monitoring_count)
    ax.set_xlabel('t')
    ax.plot(plot_x, ep_stats['avgPacketDelay'], label="avg. delay")
    fig.canvas.draw()
    ep_plot_dict['avg_delay'] = Image.frombytes('RGB',
                                                fig.canvas.get_width_height(),
                                                fig.canvas.tostring_rgb()
                                                )
    plt.close()

    # max LU and bandwidth usage per timestep: 'bw_lu'
    fig, ax = plt.subplots()
    ax.set_title(f"{ep_name}: max LU/total bandwidth usage")
    ax.set_xlim(1, ep_monitoring_count)
    ax.set_xlabel('t')
    ax.plot(plot_x, ep_stats['maxLU'], label="max. LU")
    ax.plot(plot_x, ep_stats['avgTDU'], label="Total datarate usage ratio")
    ax.legend()
    fig.canvas.draw()
    ep_plot_dict['bw_lu'] = Image.frombytes('RGB',
                                            fig.canvas.get_width_height(),
                                            fig.canvas.tostring_rgb()
                                            )
    plt.close()

    # reward over time
    fig, ax = plt.subplots()
    ax.set_title(f"{ep_name}: global reward")
    ax.set_xlim(1, ep_monitoring_count)
    ax.set_xlabel('t')
    ax.plot(plot_x, ep_stats['global_rewards'], label="global reward")
    ax.legend()
    fig.canvas.draw()
    ep_plot_dict['global_rewards'] = Image.frombytes('RGB',
                                                     fig.canvas.get_width_height(),
                                                     fig.canvas.tostring_rgb()
                                                     )
    plt.close()

    plt.close('all')
    return ep_plot_dict
