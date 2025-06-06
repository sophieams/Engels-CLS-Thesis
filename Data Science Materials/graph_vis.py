"""
Simple script for visualizing the extracted graphs from the MTGNN.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Plotting preferences
plt.rcParams.update({
    "font.family": "serif",
    "font.size":24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

def visualize_adjacency_matrix(A_tilde):
    """
    Function to visualize the graphs from the MTGNN with NetworkX.
    Input: 
    - A_tilde: the graph from the MTGNN we want to visualize as a numpy array.
    Returns nothing.
    """
    A_tilde = np.array(A_tilde)
    G = nx.from_numpy_array(A_tilde, create_using=nx.DiGraph)
    
    symptom_labels = ['Energy', 'Valence', 'Calmness', 'PA', 'Anxiety', 
                  'Depression', 'PB', 'TB', 'Hopelessness', 'SI']

    for i, label in enumerate(symptom_labels):
        G = nx.relabel_nodes(G, {i: label})

    pos = nx.spring_layout(G, seed=42, k=5)
    
    _, ax = plt.subplots(figsize=(15, 7.5)) 
    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color="skyblue",
        node_size=5500,
        font_size=14,
        arrows=True,
        edge_color='darkblue',
        edge_cmap=plt.cm.Spectral,
        connectionstyle='arc3,rad=0.15',
        arrowsize=20,
        width=2
    )

    ax.set_title("Learned Graph from MTGNN for Individual 2")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
learned_A_tilde_1 = [
    [0., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 1., 0.9999972, 0., 0.],
    [0., 1., 1., 0., 1., 0., 1., 1., 1., 0.9971849],
    [0., 1., 1., 0., 0., 0., 1., 1., 0., 0.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 1., 0., 0.9966562, 0., 1., 1., 0., 0.],
    [0., 1., 1., 0., 1., 0., 1., 1., 1., 0.]
]

visualize_adjacency_matrix(learned_A_tilde_1)

