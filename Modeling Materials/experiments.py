"""
This script runs various experiments on the extended Wang model.
The experiments include parameter sweeps, simulations of different social configurations, 
different network densities, rewiring probabilities, aversive internal state offsets for the neighbors,
and Barabasi-Albert configurations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, mannwhitneyu
from wang_model import set_params
from network_suicide import create_network_ws, create_network_ba, run_simulation

# plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})

initial_params_names = ["S0", "A0", "U0", "T0", "O0", "E0", "I0", "PB0"]
other_params_names = ["sigma", "f1", "K2", "b2", "a2", "d2", "e2", "g2", "c3", "b3", "d4", "c41", "c42", 
                        "e5", "c51", "c52", "K6", "f6", "b6", "c6", "K7", "g7", "b7", "c7", 
                        "a3",  "h2", "b8", "c8", "c9"]

param_idx = {}
for i, name in enumerate(other_params_names):
    param_idx[name] = i

def set_params_experiment(mode, override_dict=None):
    """
    This function sets the initial and other parameters for the Wang model based on the specified mode.
    The mode can be "low", "medium", or "high" to adjust the stressor settings.
    If override_dict is provided, it will change the corresponding parameters for the next simulation.
    Input:
    mode: str - "low", "medium", or "high" to set the stressor settings.
    override_dict: dict - Optional dictionary to override specific parameters.
    Output:
    init_params: np.array - Initial parameters for the Wang model.
    other_params: tuple - Other parameters for the Wang model from wang_model.py.
    """
    init_params, other_params = set_params(mode)
    if override_dict:
        other_params = list(other_params)
        for name, value in override_dict.items():
            idx = param_idx[name]
            other_params[idx] = value
    return init_params, tuple(other_params)

def simulate_with_sweep(param_name, param_values, t, dt, num_simulations, patient_node=0):
    """
    This function runs simulations for a range of parameter values and collects results.
    Input:
    param_name: The name of the parameter to sweep (e.g., "a3", "h2", etc.).
    param_values: The values of the parameter to test (list).
    t: Total number of minutes for the simulation.
    dt: Time step for the simulation.
    num_simulations: Number of simulations to run for each parameter value.
    patient_node: The node number of the patient in the network.
    Output:
    results: A list of dictionaries containing the results of each simulation.
    Each dictionary contains the simulation number, parameter value, and maximum values of various states.
    """
    results = []

    for val in param_values:
        for sim in range(num_simulations):
            print(f"Testing {param_name} = {val} | Simulation {sim}")

            override = {param_name: val} if param_name not in ["temperature"] else {}

            G = create_network_ws(
                N=15, k=15, patient_node=patient_node, t=t, dt=dt,
                strong_ties=list(range(1, 3)), medium_ties=list(range(3, 8)),
                seed_number=504 + sim
            )

            if param_name != "temperature":
                init, other = set_params_experiment("high", override)
                G.nodes[patient_node]["initial_params"] = init
                G.nodes[patient_node]["other_params"] = other

            temperature = val if param_name == "temperature" else 0.3
            run_simulation(G, t, dt, patient_node, temperature=temperature)

            state = G.nodes[patient_node]["variable"]
            results.append({
                "simulation_number": sim,
                "param_value": val,
                "max_av_state": np.max(state[1]),
                "max_urge_escape": np.max(state[2]),
                "max_suicidal_thoughts": np.max(state[3]),
                "max_escape_behaviors": np.max(state[4]),
                "max_ext_change": np.max(state[5]),
                "max_int_change": np.max(state[6]),
                "max_pb": np.max(state[7])
            })

    return results
        
def simulate_social_configurations_multiple_seeds():
    """
    This function simulates different social configurations with varying numbers of strong and medium ties.
    It runs 10 simulations for each configuration and saves the results to CSV files.
    Each configuration is defined by the number of nodes (N), edges (k), and the presence of strong and medium ties.
    The results include the mean and standard error of various states over time.
    The configurations are:
    1. 15 nodes with strong ties (1-2) and medium ties (3-7).
    2. 50 nodes with no strong or medium ties.
    3. 3 nodes with strong ties (1-2) and no medium ties.
    4. 1 node with no strong or medium ties.
    The results are saved in the "social_configurations" directory.
    """
    configuration_1 = [
        {"name": "N", "values": 15, "label": "N"},
        {"name": "k", "values": 15, "label": "k"},
        {"name": "strong_ties", "values": list(range(1, 3)), "label": "Strong_Ties"},
        {"name": "medium_ties", "values": list(range(3, 8)), "label": "Medium_Ties"},
    ]
    
    configuration_2 = [
        {"name": "N", "values": 50, "label": "N"},
        {"name": "k", "values": 15, "label": "k"},
        {"name": "strong_ties", "values": None, "label": "No_Strong_Ties"},
        {"name": "medium_ties", "values": None, "label": "No_Medium_Ties"},
    ]
    
    configuration_3 = [
        {"name": "N", "values": 3, "label": "N"},
        {"name": "k", "values": 2, "label": "k"},
        {"name": "strong_ties", "values": list(range(1, 3)), "label": "Strong_Ties"},
        {"name": "medium_ties", "values": None, "label": "No Medium_Ties"},
    ]
    
    configuration_4 = [
        {"name": "N", "values": 1, "label": "N"},
        {"name": "k", "values": 0, "label": "k"},
        {"name": "strong_ties", "values": None, "label": "No_Strong_Ties"},
        {"name": "medium_ties", "values": None, "label": "No_Medium_Ties_or_Weak_Ties"},
    ]

    configurations = [configuration_1, configuration_2, configuration_3, configuration_4]
    for config in configurations:
        all_results = []
        for sim in range(10):
            G = create_network_ws(
                N=config[0]["values"],
                k=config[1]["values"],
                patient_node=0,
                t=20160,
                dt=0.01,
                strong_ties=config[2]["values"],
                medium_ties=config[3]["values"],
                seed_number=504 + sim
            )
            init, other = set_params_experiment("high")
            G.nodes[0]["initial_params"] = init
            G.nodes[0]["other_params"] = other
            run_simulation(G, t=20160, dt=0.01, patient_node=0, temperature=0.3)
            state = G.nodes[0]["variable"]
            all_results.append(state)

        all_results = np.array(all_results) 
        mean_results = np.mean(all_results, axis=0) 
        sem_results = np.std(all_results, axis=0) / np.sqrt(all_results.shape[0]) 

        combined_results = []
        for i, label in enumerate(["av_state", "urge_escape", "sui_thoughts", "other_escape", 
                                    "ext_change", "int_change", "social_burden"]):
            for time_step in range(mean_results.shape[1]):
                combined_results.append({
                    "time": time_step,
                    "state": label,
                    "mean": mean_results[i, time_step],
                    "sem": sem_results[i, time_step]
                })

        df = pd.DataFrame(combined_results)
        df.to_csv(f"social_configurations/{config[2]['label']}_{config[3]['label']}_1sim.csv", index=False)
        print(f"Configuration {config[2]['label']}_{config[3]['label']} completed.")

def simulate_different_densities_multiple_seeds():
    """
    This function simulates different network densities by varying the number of edges (k) in a Watts-Strogatz network.
    It runs 20 simulations for each density configuration (k = 5, 10, 14) and saves the results to CSV files.
    The results include the mean and standard error of various states over time.
    """
    k_values = [5, 10, 14]
    for k in k_values:
        all_results = []  
        for sim in range(20):
            G = create_network_ws(
                N=15,
                k=k,
                patient_node=0,
                t=20160,
                dt=0.01,
                strong_ties=None,
                medium_ties=None,
                seed_number=504 + sim
            )
            init, other = set_params_experiment("high")
            G.nodes[0]["initial_params"] = init
            G.nodes[0]["other_params"] = other
            run_simulation(G, t=20160, dt=0.01, patient_node=0, temperature=0.3)
            state = G.nodes[0]["variable"]
            all_results.append(state)

        all_results = np.array(all_results)  
        mean_results = np.mean(all_results, axis=0)
        sem_results = np.std(all_results, axis=0) / np.sqrt(all_results.shape[0]) 

        combined_results = []
        for i, label in enumerate(["av_state", "urge_escape", "sui_thoughts", "other_escape", 
                                    "ext_change", "int_change", "social_burden"]):
            for time_step in range(mean_results.shape[1]):
                combined_results.append({
                    "time": time_step,
                    "state": label,
                    "mean": mean_results[i, time_step],
                    "sem": sem_results[i, time_step]
                })

        df = pd.DataFrame(combined_results)
        df.to_csv(f"social_configurations/cc_{k}_1sim.csv", index=False)
        print(f"Configuration {k} completed.")
            
def simulate_different_rewiring_probabilities_multiple_seeds():
    """
    This function simulates different rewiring probabilities in a Watts-Strogatz network.
    It runs 20 simulations for each rewiring probability (0.01, 0.1, 1) and saves the results to CSV files.
    The results include the mean and standard error of various states over time.
    """
    rewiring_probabilities = [0.01, 0.1, 1]
    for p in rewiring_probabilities:
        all_results = []
        for sim in range(20):
            G = create_network_ws(
                N=150,
                k=50,
                patient_node=0,
                t=20160,
                dt=0.01,
                strong_ties=None,
                medium_ties=None,
                seed_number=504 + sim,
                p = p
            )
            init, other = set_params_experiment("high")
            G.nodes[0]["initial_params"] = init
            G.nodes[0]["other_params"] = other
            run_simulation(G, t=20160, dt=0.01, patient_node=0, temperature=p)
            state = G.nodes[0]["variable"]
            all_results.append(state)
            
        all_results = np.array(all_results) 
        mean_results = np.mean(all_results, axis=0)
        sem_results = np.std(all_results, axis=0) / np.sqrt(all_results.shape[0]) 

        combined_results = []
        for i, label in enumerate(["av_state", "urge_escape", "sui_thoughts", "other_escape", 
                                    "ext_change", "int_change", "social_burden"]):
            for time_step in range(mean_results.shape[1]):
                combined_results.append({
                    "time": time_step,
                    "state": label,
                    "mean": mean_results[i, time_step],
                    "sem": sem_results[i, time_step]
                })

        df = pd.DataFrame(combined_results)
        df.to_csv(f"social_configurations/new_{p}_rw_prob.csv", index=False)
        print(f"Configuration {p} completed.")
    
def simulate_different_ais_offsets_multiple_seeds():
    """
    This function simulates different AIS offsets in a Watts-Strogatz network.
    It runs 20 simulations for each AIS offset (0.0, 0.3, 0.5) and saves the results to CSV files.
    The results include the mean and standard error of various states over time.
    """
    ais_offsets = [0.0, 0.3, 0.5]
    for ais_offset in ais_offsets:
        all_results = []
        for sim in range(20):
            G = create_network_ws(
                N=15,
                k=14,
                patient_node=0,
                t=20160,
                dt=0.01,
                strong_ties=None,
                medium_ties=None,
                seed_number=504 + sim
            )
            init, other = set_params_experiment("high")
            G.nodes[0]["initial_params"] = init
            G.nodes[0]["other_params"] = other
            run_simulation(G, t=20160, dt=0.01, patient_node=0, temperature=0.3, ais_offset=ais_offset)
            state = G.nodes[0]["variable"]
            all_results.append(state)

        all_results = np.array(all_results) 
        mean_results = np.mean(all_results, axis=0)
        sem_results = np.std(all_results, axis=0) / np.sqrt(all_results.shape[0]) 
        combined_results = []
        for i, label in enumerate(["av_state", "urge_escape", "sui_thoughts", "other_escape", 
                                    "ext_change", "int_change", "social_burden"]):
            for time_step in range(mean_results.shape[1]):
                combined_results.append({
                    "time": time_step,
                    "state": label,
                    "mean": mean_results[i, time_step],
                    "sem": sem_results[i, time_step]
                })

        df = pd.DataFrame(combined_results)
        df.to_csv(f"social_configurations/{ais_offset}_ais_offset.csv", index=False)
        print(f"Configuration {ais_offset} completed.")
        
def simulate_different_ba_configs_multiple_seeds():
    """
    This function simulates different Barabasi-Albert configurations.
    It runs 10 simulations for each configuration (patient and non-patient) and saves the results to CSV files.
    The results include the mean and standard error of various states over time.
    """
    ba_configs = ["patient", "nonpatient"]
    for config in ba_configs:
        all_results = []
        for sim in range(10):
            G, patient_node = create_network_ba(
                N=150,
                m=1,
                patient_node=0,
                t=20160,
                dt=0.01,
                strong_ties=None,
                medium_ties=None,
                seed_number=504 + sim, 
                hub_role=config
            )
            init, other = set_params_experiment("high")
            G.nodes[0]["initial_params"] = init
            G.nodes[0]["other_params"] = other
            run_simulation(G, t=20160, dt=0.01, patient_node=patient_node, temperature=0.3)
            state = G.nodes[0]["variable"]
            all_results.append(state)

        all_results = np.array(all_results)
        mean_results = np.mean(all_results, axis=0) 
        sem_results = np.std(all_results, axis=0) / np.sqrt(all_results.shape[0]) 
        combined_results = []
        for i, label in enumerate(["av_state", "urge_escape", "sui_thoughts", "other_escape", 
                                    "ext_change", "int_change", "social_burden"]):
            for time_step in range(mean_results.shape[1]):
                combined_results.append({
                    "time": time_step,
                    "state": label,
                    "mean": mean_results[i, time_step],
                    "sem": sem_results[i, time_step]
                })

        df = pd.DataFrame(combined_results)
        df.to_csv(f"social_configurations/{config}_ba_config.csv", index=False)
        print(f"Configuration {config} completed.")

def plot_stacked_time_series(
    data_files,
    labels,
    output_file,
    legend_cols=3,
    states=["other_escape", "social_burden", "sui_thoughts"],
    state_titles=["Escape Behavior", "Social Burden", "Suicidal Thoughts"],
    skip_initial=2880,
    x_ticks=np.arange(0, 13, 1), 
    suptitle=None
):
    """
    This function plots stacked time series for different states from multiple CSV files.
    Input:
    data_files: List of CSV files containing the simulation results.
    labels: List of labels for each data file.
    output_file: The name of the output file to save the plot.
    legend_cols: Number of columns in the legend.
    states: List of states to plot.
    state_titles: Titles for each state plot.
    skip_initial: Number of initial time steps to skip for each state.
    x_ticks: Ticks for the x-axis.
    suptitle: Optional title for the entire figure.
    Output:
    A stacked time series plot saved to the specified output file.
    """
    fig, axes = plt.subplots(len(states), 1, figsize=(15, 15), sharex=True)

    for ax, state, title in zip(axes, states, state_titles):
        for file, label in zip(data_files, labels):
            df = pd.read_csv(file)
            df = df[df["state"] == state]
            time = np.arange(len(df["mean"][skip_initial:])) / 1440
            mean = df["mean"][skip_initial:]
            sem = df["sem"][skip_initial:]

            ax.plot(time, mean, label=label)
            ax.fill_between(time, mean - sem, mean + sem, alpha=0.2)

        ax.set_xticks(x_ticks)
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        ax.grid(True)

    axes[-1].set_xlabel("Days")
    axes[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=legend_cols)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=30, y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file)
    
def plot_sweep_results_stacked(param_sweeps):
    """
    This function plots the results of parameter sweeps in a stacked format.
    It reads the results from CSV files generated by the simulations and plots the maximum values of various states
    for each parameter sweep.
    Input:
    param_sweeps: List of dictionaries containing parameter sweep information.
    Output:
    A stacked plot showing the maximum values of various states for each parameter sweep.
    """
    param_names = [sweep["name"] for sweep in param_sweeps]
    
    _, axes = plt.subplots(len(param_names), 1, figsize=(15, 30), sharex=False)
    for ax, param_name in zip(axes, param_names):
        df = pd.read_csv(f"sweep_results/{param_name}_sweep.csv")
        grouped = df.groupby("param_value")

        means = grouped.mean()
        errors = grouped.sem()

        x = means.index.values

        for col, label in [
            ("max_av_state", "Max Aversive Internal State"),
            ("max_suicidal_thoughts", "Max Suicidal Thoughts"),
            ("max_escape_behaviors", "Max Escape Behaviors"),
            ("max_pb", "Max Social Burden")
        ]:
            y = means[col].values
            yerr = errors[col].values

            ax.plot(x, y, label=label, marker="o")
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        label = next((sweep["label"] for sweep in param_sweeps if sweep["name"] == param_name), param_name)
        ax.set_ylabel("Max Value")
        if param_name == "a3":
            ax.set_xlabel("a3 (Effect of Social Burden on AIS)")  
            ax.set_title(f"Maximum State Values for Varying a3")
        if param_name == "h2":
            ax.set_xlabel("h2 (Effect of Neighbor AIS on Social Burden)")
            ax.set_title(f"Maximum State Values for Varying h2")
        if param_name == "b8":
            ax.set_xlabel("b8 (Effect of Internally Focused Change on Social Burden)")  
            ax.set_title(f"Maximum State Values for Varying b8")
        if param_name == "c8":
            ax.set_xlabel("c8 (Natural Decay of Social Burden)")
            ax.set_title(f"Maximum State Values for Varying c8")
        if param_name == "c9":
            ax.set_xlabel("c9 (Effect of Social Connectedness on Urge to Escape)")
            ax.set_title(f"Maximum State Values for Varying c9")
        if param_name == "temperature":
            ax.set_xlabel("Reliance Distribution")
            ax.set_title(f"Maximum State Values for Varying Reliance Distribution")
        ax.grid(True)
        ax.set_xlim(x[0], x[-1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol = 2)
    plt.subplots_adjust(bottom=0.25)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/stacked_sweep_plot.png')
    
def calculate_stats():
    """
    This function calculates statistics for the mean output distributions in patient and non-patient Barabasi-Albert configurations.
    It tests for normality of distributions using the Kolmogorov-Smirnov test and compares distributions using the Mann-Whitney U test.
    It then plots the distributions of various states (other_escape, social_burden, sui_thoughts) for both configurations.
    """
    data_files = [
        "social_configurations/patient_ba_config.csv",
        "social_configurations/nonpatient_ba_config.csv"
    ]
    # test normality of distributions of various states
    for file in data_files:
        df = pd.read_csv(file)
        for state in ["other_escape", "social_burden", "sui_thoughts"]:
            state_df = df[df["state"] == state]
            _, p = kstest(state_df["mean"], 'norm')
            if p < 0.05:
                print(f"{state} in {file} is not normally distributed.")
            else:
                print(f"{state} in {file} is normally distributed.")
        print()
        
    # test for differences between distributions of various states
    df1 = pd.read_csv(data_files[0])
    df2 = pd.read_csv(data_files[1])
    for state in ["other_escape", "social_burden", "sui_thoughts"]:
        state_df1 = df1[df1["state"] == state]
        state_df2 = df2[df2["state"] == state]
        u_stat, p_value = mannwhitneyu(state_df1["mean"], state_df2["mean"])
        print(f"Mann-Whitney U test for {state}: U-statistic = {u_stat}, p-value = {p_value}")
    print()
    
    #plot the distributions
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    for ax, state in zip(axes, ["other_escape", "social_burden", "sui_thoughts"]):
        for file in data_files:
            df = pd.read_csv(file)
            state_df = df[df["state"] == state]
            ax.hist(state_df["mean"], bins=30, alpha=0.5, label=f"{state} in {file.split('/')[-1].split('.')[0]}")
        ax.set_ylabel("Frequency")
        ax.grid(True)
    axes[2].set_xlabel("Value")
    axes[0].set_title(f"Distribution of Escape Behavior")
    axes[1].set_title(f"Distribution of Social Burden")
    axes[2].set_title(f"Distribution of Suicidal Thoughts")
    plt.legend(labels=["Hub Patient Data", "Remote Patient Data"], loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol = 2)
    fig.suptitle("Mean Output Distributions in Patient BA Configurations", fontsize=30, y=0.97)
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.savefig("plots/distribution_combined.png")
    plt.close()
    
# Calculate statistics for the mean output distributions 
calculate_stats()

# Parameter Sweeps
param_sweeps = [
    {"name": "a3",
    "values": np.linspace(0, 2, 10),
    "label": "a3 (Effect of Social Burden on AIS)"},
    {"name": "h2",
    "values": np.linspace(0, 2, 10),
    "label": "h2 (Effect of Neighbor AIS on Social Burden)"},
    {"name": "b8",
    "values": np.linspace(0, 2, 10),
    "label": "b8 (Effect of Internally Focused Change on Social Burden)"},
    {"name": "c8",
    "values": np.linspace(0.1, 1, 10),
    "label": "c8 (Natural Decay of Social Burden)"},
    {"name": "c9",
    "values": np.linspace(0, 1, 10),
    "label": "c9 (Effect of Social Connectedness on Urge to Escape)"},
    {"name": "temperature",
    "values": np.linspace(0.1, 1, 10),
    "label": "temperature (Weight Sensitivity in Social Connectedness)"}
]

# Simulations
for sweep in param_sweeps:
    results = simulate_with_sweep(
        sweep["name"],
        sweep["values"],
        t=20160,
        dt=0.01,
        num_simulations=10
    )
    df = pd.DataFrame(results)
    df.to_csv(f"sweep_results/{sweep['name']}_sweep.csv", index=False)
    print()
plot_sweep_results_stacked(param_sweeps)   
simulate_social_configurations_multiple_seeds()
simulate_different_densities_multiple_seeds()
simulate_different_rewiring_probabilities_multiple_seeds()
simulate_different_ais_offsets_multiple_seeds()
simulate_different_ba_configs_multiple_seeds()

# Plotting Data
# BA Configurations
plot_stacked_time_series(
    data_files=[
        "social_configurations/patient_ba_config.csv",
        "social_configurations/nonpatient_ba_config.csv"
    ],
    labels=["Hub Patient", "Remote Patient"],
    output_file="plots/stacked_ba_plots.png",
    suptitle="Barabasi-Albert Graph (150 Nodes, 1 Edge Added per Node)"
)

# # AIS Offsets
plot_stacked_time_series(
    data_files=[
        "social_configurations/0.0_ais_offset.csv",
        "social_configurations/0.3_ais_offset.csv",
        "social_configurations/0.5_ais_offset.csv"
    ],
    labels=["0 AIS Offset", "0.3 AIS Offset", "0.5 AIS Offset"],
    output_file="plots/stacked_ais_plots.png", 
    suptitle="Varying Neighbor AIS (15 Nodes, Fully Connected)"
)

# # Rewiring Probabilities
plot_stacked_time_series(
    data_files=[
        "social_configurations/new_0.01_rw_prob.csv",
        "social_configurations/new_0.1_rw_prob.csv",
        "social_configurations/new_1_rw_prob.csv"
    ],
    labels=["p = 0.01", "p = 0.1", "p = 1"],
    output_file="plots/stacked_rw_plots.png", 
    suptitle="Varying Rewiring Probability (150 Nodes, 50 Edges/Node)"
)

# # Social Configurations
plot_stacked_time_series(
    data_files=[
        "social_configurations/Strong_Ties_Medium_Ties_1sim.csv",
        "social_configurations/Strong_Ties_No_Medium_Ties_1sim.csv",
        "social_configurations/No_Strong_Ties_No_Medium_Ties_1sim.csv",
        "social_configurations/No_Strong_Ties_No_Medium_Ties_or_Weak_Ties_1sim.csv"
    ],
    labels=[
        "2 Strong Ties, 5 Medium Ties, 7 Weak Ties",
        "Only 2 Strong Ties",
        "15 Weak Ties",
        "No Ties"
    ],
    output_file="plots/stacked_social_configurations_plots.png",
    legend_cols=2, 
    suptitle="Varying Social Configurations (Each Graph is Fully Connected)"
)

# # Densities
plot_stacked_time_series(
    data_files=[
        "social_configurations/cc_5_1sim.csv",
        "social_configurations/cc_10_1sim.csv",
        "social_configurations/cc_14_1sim.csv"
    ],
    labels=["k = 5", "k = 10", "k = 14"],
    output_file="plots/stacked_density_plots.png", 
    suptitle="Varying Densities (15 Nodes, 5-14 Edges/Node)"
)

