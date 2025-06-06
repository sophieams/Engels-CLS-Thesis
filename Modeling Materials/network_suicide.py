"""
This script creates the social network configurations necessary for running the simulation.
Here we also calculate the contagious social burden and social connectedness.
Finally, we run the simulation from wang_model and we can visualize the social configurations
and the patient/nonpatient 12 days runs.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from wang_model import set_params, gen_suicide_step

# Usual plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

def create_network_ws(N, k, patient_node, t, dt, strong_ties, medium_ties, seed_number=504, p=0.01):
    """
    This function creates teh Newmann Watts-Strogatz social networks as the framework for the simulation.
    Input:
    - N: The number of nodes in the network
    - k: The number of connections per node.
    - patient_node: The number/key of the node that we want to assign as the patient.
    - t: The number of minutes (length of time) we want to simulate for.
    - dt: the difference between each time step that we want to numerically solve.
    - strong_ties: A list of nodes that are considered strong ties to the patient.
    - medium_ties: A list of nodes that are considered medium ties to the patient.
    - seed_number: The seed number for reproducibility.
    - p: The probability of rewiring edges.
    Output:
    - G: The graph representing the social network.
    - patient_node: The node that is assigned as the patient (so it can be reassigned later).
    """
    # Using Newman-Watts-Strogatz to generate a small-world network
    G = nx.newman_watts_strogatz_graph(n=N, k=k, p=p, seed=seed_number)

    # Assign edge strengths
    for u, v in G.edges():
        if patient_node in [u, v]:
            other = v if u == patient_node else u
            if strong_ties is not None and other in strong_ties:
                G.edges[u, v]["strength"] = np.random.uniform(0.99, 1.0)
            elif medium_ties is not None and other in medium_ties:
                G.edges[u, v]["strength"] = np.random.uniform(0.4, 0.6)
            else:
                G.edges[u, v]["strength"] = np.random.uniform(0.05, 0.1)
        else:
            G.edges[u, v]["strength"] = np.random.uniform(0.05, 0.1)
    
    # Calculate weighted clustering coefficients
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            avg_weight = np.mean([G.edges[node, neighbor]["strength"] for neighbor in neighbors])
            clustering = nx.clustering(G, nodes=node, weight="strength")
            weighted_clustering = clustering * avg_weight
        else:
            weighted_clustering = 0.0

        G.nodes[node]["clustering_coefficient"] = weighted_clustering

    for node in G.nodes():
        if node == patient_node:
            params = set_params("high")
            is_patient = True
        else:
            params = set_params("medium")
            is_patient = False

        seed = seed_number + node
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, np.sqrt(dt), size=t - 1)

        G.nodes[node]["initial_params"] = params[0]
        G.nodes[node]["other_params"] = params[1]
        G.nodes[node]["noise"] = noise
        G.nodes[node]["is_patient"] = is_patient
        G.nodes[node]["variable"] = None

    return G

def create_network_ba(N, m, patient_node, t, dt, strong_ties, medium_ties, seed_number=504, hub_role="patient"):
    """    
    This function creates a Barabási-Albert network as the framework for the simulation.
    This is very similar to the previous function, but it uses a different network generation method.
    Input:
    - N: The number of nodes in the network.
    - m: The number of edges to attach from a new node to nodes already added to the network.
    - patient_node: The number/key of the node that we want to assign as the patient.
    - t: The number of minutes (length of time) we want to simulate for.
    - dt: The difference between each time step that we want to numerically solve.
    - strong_ties: A list of nodes that are considered strong ties to the patient.
    - medium_ties: A list of nodes that are considered medium ties to the patient.
    - seed_number: The seed number for reproducibility.
    - hub_role: The role of the hub node, either "patient" or "nonpatient".
    Output:
    - G: The graph representing the social network.
    - patient_node: The node that is assigned as the patient (so it can be reassigned later).
    """
    # Generate a Barabási-Albert network
    G = nx.barabasi_albert_graph(n=N, m=m, seed=seed_number)

    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    largest_hub = hubs[0][0]
    shortest_paths = nx.single_source_shortest_path_length(G, largest_hub)
    most_remote_node = max(shortest_paths, key=shortest_paths.get)
    smallest_degree_node = most_remote_node

    # Assign the hub role
    if hub_role == "patient":
        patient_node = largest_hub
    elif hub_role == "nonpatient":
        patient_node = smallest_degree_node

    # Same as before
    for u, v in G.edges():
        if patient_node in [u, v]:
            other = v if u == patient_node else u
            if strong_ties is not None and other in strong_ties:
                G.edges[u, v]["strength"] = np.random.uniform(0.99, 1.0)
            elif medium_ties is not None and other in medium_ties:
                G.edges[u, v]["strength"] = np.random.uniform(0.4, 0.6)
            else:
                G.edges[u, v]["strength"] = np.random.uniform(0.05, 0.1)
        else:
            G.edges[u, v]["strength"] = np.random.uniform(0.05, 0.1)

    # Same as before
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            avg_weight = np.mean([G.edges[node, neighbor]["strength"] for neighbor in neighbors])
            clustering = nx.clustering(G, nodes=node, weight="strength")
            weighted_clustering = clustering * avg_weight
        else:
            weighted_clustering = 0.0

        G.nodes[node]["clustering_coefficient"] = weighted_clustering

    for node in G.nodes():
        if node == patient_node:
            params = set_params("high")
            is_patient = True
        else:
            params = set_params("medium")
            is_patient = False

        seed = seed_number + node
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, np.sqrt(dt), size=t - 1)

        G.nodes[node]["initial_params"] = params[0]
        G.nodes[node]["other_params"] = params[1]
        G.nodes[node]["noise"] = noise
        G.nodes[node]["is_patient"] = is_patient
        G.nodes[node]["variable"] = None

    return G, patient_node

def compute_social_burden(G, time_step, node, ais_offset=0):
    """
    Computes the social burden for a node at a specific time step.
    The social burden is the sum of the aversive internal states of the neighbors,
    adjusted by a constant offset that can be manipulated in an experiment.
    Parameters:
    - G: The graph.
    - time_step: The current time step in the simulation.
    - node: The node for which to compute the social burden.
    - ais_offset: A constant to add to the aversive internal state of neighbors.
    Returns:
    - social_sum: The computed social burden for the node.
    """
    if len(list(G.neighbors(node))) == 0:
        return 0.0
    
    social_sum = 0.0
    for neighbor in G.neighbors(node):
        neighbor_state = G.nodes[neighbor]["variable"]
        neighbor_ais = neighbor_state[1, time_step] + ais_offset
        weight = G.edges[node, neighbor].get("strength", 1.0)
        social_sum += neighbor_ais * weight

    return social_sum

def run_simulation(G, t, dt, temperature, ais_offset=0.0):
    """
    Runs the simulation for all nodes in the graph.
    Input:
    - G: The graph.
    - t: Total time steps.
    - dt: Time step size.
    - patient_node: The patient node.
    - temperature: Reliance distribution parameter for social connectedness.
    - ais_offset: A constant to add to the aversive internal state of neighbors.
    No output, but updates the variable attribute of each node.
    """
    # Initialize states for all nodes
    for node in G.nodes():
        init = G.nodes[node]["initial_params"]
        other = G.nodes[node]["other_params"]

        G.nodes[node]["variable"] = np.zeros((8, t))
        G.nodes[node]["variable"][0, 0] = init[0]  # stressor
        G.nodes[node]["variable"][1, 0] = init[1]  # AIS
        G.nodes[node]["variable"][2, 0] = init[2]  # urge to escape
        G.nodes[node]["variable"][3, 0] = init[3]  # suicidal thoughts
        G.nodes[node]["variable"][4, 0] = init[4]  # escape behaviors
        G.nodes[node]["variable"][5, 0] = init[5]  # external-focused change
        G.nodes[node]["variable"][6, 0] = init[6]  # internal-focused change
        G.nodes[node]["variable"][7, 0] = init[7]  # social burden

    # Iteratively update variables for all nodes
    for time_step in range(t - 1):  # Must go to t-1 because we simulate the next step
        for node in G.nodes():
            neighbor_burden_val = compute_social_burden(G, time_step, node, ais_offset=ais_offset)
            social_connectedness = compute_social_connectedness(G, node, temperature)

            current_state = G.nodes[node]["variable"][:, time_step].copy()
            other = G.nodes[node]["other_params"]
            previous_noise = G.nodes[node]["noise"][time_step]

            next_state = gen_suicide_step(
                current_state, dt, other, previous_noise,
                neighbor_burden_val, social_connectedness,
                G.nodes[node].get("clustering_coefficient", 0.0)
            )

            G.nodes[node]["variable"][:, time_step + 1] = next_state

def compute_social_connectedness(G, patient_node, temperature):
    """
    This function calculates the social connectedness and reliance distribution of a particular node.
    Input:
    - G: the graph
    - patient_node: the number/key of the node delineated as the patient.
    - temperature: the reliance distribution parameter (lower means higher reliance on close friends).
    Output:
    - connectedness: a proxy for social connectedness using an entropy calculation of the edge weights.
    """
    if len(list(G.neighbors(patient_node))) == 0:
        return 0.0
    
    weights = []
    for neighbor in G.neighbors(patient_node):
        weights.append(G.edges[patient_node, neighbor].get("strength", 1.0))

    weights = np.array(weights) 
    softmax_weights = np.exp(weights / temperature)
    softmax_weights /= softmax_weights.sum()

    # Entropy: higher = more evenly distributed, lower = more weight on close friends
    entropy = -np.sum(softmax_weights * np.log(softmax_weights + 1e-8))

    max_entropy = np.log(len(softmax_weights))
    normalized_entropy = entropy / max_entropy
    connectedness = 1 - normalized_entropy 

    return connectedness

def plot_trajectories(G, patient_node):
    """
    This function plots the trajectories of the patient and a non-patient node.
    Input:
    - G: The graph.
    - patient_node: The node that is assigned as the patient.
    Output:
    - Saves a plot of the trajectories of the patient and a non-patient node in the folder 'plots'.
    """
    time_days = np.arange(G.nodes[patient_node]["variable"].shape[1]) / 1440
    patient_states = G.nodes[patient_node]["variable"]

    # Pick the next node as a nonpatient example
    for n in G.nodes:
        if not G.nodes[n]["is_patient"]:
            non_patient_node = n
            break
    non_patient_states = G.nodes[non_patient_node]["variable"]

    symptoms = [
        "Stressor", "Aversive Internal State", "Urge to Escape",
        "Suicidal Thoughts", "Escape Behavior",
        "External-focused Change", "Internal-focused Change",
        "Social Burden"
    ]

    _, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True)

    # Patient trajectory
    day_vals = time_days[2880:] - time_days[2880]
    for i in range(len(symptoms)):
        axes[0].plot(day_vals, patient_states[i][2880:], label=symptoms[i])
    axes[0].set_title("Patient Node Example Simulation")
    axes[0].set_ylabel("Intensity")
    axes[0].grid()

    # Non-patient trajectory
    for i in range(len(symptoms)):
        axes[1].plot(day_vals, non_patient_states[i][2880:], label=symptoms[i])
    axes[1].set_title(f"Non-Patient Node Example Simulation")
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Intensity")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[1].grid()

    plt.xticks(ticks=np.arange(0, 13, step=1))
    plt.tight_layout()
    plt.savefig(f"plots/example_run.png", bbox_inches='tight')
    
def visualize_network(G, patient_node):
    """
    This function visualizes the network with the patient node in the center.
    Input:
    - G: The graph.
    - patient_node: The node that is assigned as the patient.
    Output:
    - Saves a plot of the network in the folder 'plots'.
    """
    # Circular layout to put the patient in the middle
    edge_nodes = list(set(G.nodes()) - {patient_node})
    pos = nx.circular_layout(G.subgraph(edge_nodes))
    # For larger BA networks, the spring layout can be used instead
    # pos = nx.spring_layout(G.subgraph(edge_nodes), seed=45)
    pos[patient_node] = np.array([0.0, 0.0])

    node_colors = []
    for node in G.nodes():
        if node == patient_node:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')

    edge_weights = []
    for u, v in G.edges():
        edge_weights.append(G.edges[u, v]["strength"])

    edge_widths = []
    for w in edge_weights:
        edge_widths.append(2 * w)
        
    plt.figure(figsize=(15, 7.5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150)
    nx.draw_networkx_edges(G, pos, edge_color=edge_weights, width=edge_widths, edge_cmap=plt.cm.Blues)
    
    # Relabels patient node as 'patient'
    labels = {}
    for node in G.nodes():
        if node == patient_node:
            labels[node] = "Patient"
        else:
            labels[node] = str(node)
    
    # If you want to add numbered labels, uncomment the next line
    #nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    plt.axis("off")
    plt.savefig(f"plots/1_rw_network_visualization.png", bbox_inches='tight')


# Example usage
N = 15
k = 14
t, dt = 20160, 0.01
patient_node = 1
strong_ties = list(range(1,3))
medium_ties = list(range(3,8))

G = create_network_ws(N, k, patient_node, t, dt, strong_ties=strong_ties, medium_ties=medium_ties, seed_number = 504, p=1)
visualize_network(G, patient_node=patient_node)
run_simulation(G, t, dt, patient_node=patient_node, temperature=0.3)
plot_trajectories(G, patient_node)
