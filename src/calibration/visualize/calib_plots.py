import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
from typing import List, Dict

sns.set_theme()
# sns.set_style("white")
sns.set_context('talk')
plt.style.use('seaborn-deep')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
sns.set_context('paper')
# from matplotlib import rc
# rc('font', family='serif')
plt.rcParams['text.color'] = 'black'
# rc('text', usetex=True)
#
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rc('text.latex', preamble=r"\usepackage{xcolor}")

# print([k for k in plt.rcParams.keys() if k.startswith('lines')])
import csv

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.kaleido.scope.mathjax = None

plt.tight_layout()


def save_results_conf(path):
    data = {
        "model": [],
        "dataset": [],
        "acc": [],
        "auc": [],
        "mce": []
    }

    # Add data to the dictionary
    models = ["Base", "RAG", "LLaMA", "GPT-NeoxT", "Flan-UL2"]
    datasets = ["squad_adversarial", "trivia", "hotpot"]
    acc_values = [
        [0.642, 0.644, 0.656, 0.660, 0.666],
        [0.601, 0.659, 0.669, 0.678, 0.665],
        [0.611, 0.635, 0.643, 0.630, 0.651]
    ]
    auc_values = [
        [0.685, 0.709, 0.716, 0.713, 0.749],
        [0.583, 0.688, 0.748, 0.745, 0.732],
        [0.740, 0.793, 0.800, 0.800, 0.816]
    ]

    mce_values = [
        [0.474, 0.465, 0.467, 0.473, 0.454],
        [0.539, 0.549, 0.507, 0.507, 0.520],
        [0.502, 0.495, 0.487, 0.482, 0.488]
    ]

    for model in models:
        for i, dataset in enumerate(datasets):
            data["model"].append(model)
            data["dataset"].append(dataset)
            data["acc"].extend(acc_values[i])
            data["auc"].extend(auc_values[i])
            data["mce"].extend(mce_values[i])

    # Write data to CSV file
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "dataset", "acc", "auc", "mce"])
        for i in range(len(data["model"])):
            writer.writerow([data["model"][i], data["dataset"][i], data["acc"][i], data["auc"][i], data["mce"][i]])

    print("CSV file saved successfully.")


def load_data(method, dataset):
    acc, auc, mce = [], [], []
    if method == "conf" and dataset == "squad_adversarial":
        # conf squad
        acc = [0.642, 0.644, 0.656, 0.660, 0.666]
        auc = [0.685, 0.709, 0.716, 0.713, 0.749]
        mce = [0.474, 0.465, 0.467, 0.473, 0.454]
    elif method == "conf" and dataset == "trivia_qa":
        # conf trivia
        acc = [0.601, 0.659, 0.669, 0.678, 0.665]
        auc = [0.583, 0.688, 0.748, 0.745, 0.732]
        mce = [0.539, 0.549, 0.507, 0.507, 0.520]
    elif method == "conf" and dataset == "hotpot_qa":
        # conf hotpot
        acc = [0.611, 0.635, 0.643, 0.630, 0.651]
        auc = [0.740, 0.793, 0.800, 0.800, 0.816]
        mce = [0.502, 0.495, 0.487, 0.482, 0.488]
    elif method == "shap" and dataset == "squad_adversarial":
        acc = [0.750, 0.736, 0.734, 0.775, 0.745, 0.782, 0.752, 0.791]
        auc = [0.843, 0.840, 0.840, 0.871, 0.847, 0.875, 0.854, 0.884]
        mce = [0.471, 0.469, 0.468, 0.461, 0.470, 0.461, 0.468, 0.460]
    elif method == "shap" and dataset == "trivia_qa":
        acc = [0.720, 0.707, 0.707, 0.707, 0.703, 0.702, 0.707, 0.708]
        auc = [0.718, 0.723, 0.761, 0.764, 0.755, 0.756, 0.753, 0.753]
        mce = [0.545, 0.527, 0.505, 0.506, 0.508, 0.509, 0.515, 0.517]
    elif method == "shap" and dataset == "hotpot_qa":
        acc = [0.633, 0.646, 0.653, 0.655, 0.627, 0.630, 0.653, 0.654]
        auc = [0.755, 0.797, 0.800, 0.801, 0.796, 0.797, 0.811, 0.812]
        mce = [0.504, 0.490, 0.493, 0.494, 0.493, 0.493, 0.489, 0.490]
    elif method == "sc_attn" and dataset == "squad_adversarial":
        acc = [0.657, 0.661, 0.654, 0.716, 0.670, 0.729, 0.672, 0.732]
        auc = [0.781, 0.794, 0.781, 0.845, 0.790, 0.849, 0.791, 0.858]
        mce = [0.476, 0.469, 0.475, 0.461, 0.475, 0.459, 0.479, 0.461]
    elif method == "sc_attn" and dataset == "trivia_qa":
        acc = [0.735, 0.713, 0.705, 0.705, 0.699, 0.701, 0.707, 0.708]
        auc = [0.719, 0.722, 0.763, 0.766, 0.756, 0.758, 0.752, 0.756]
        mce = [0.558, 0.539, 0.510, 0.511, 0.511, 0.511, 0.520, 0.521]
    elif method == "sc_attn" and dataset == "hotpot_qa":
        acc = [0.631, 0.641, 0.640, 0.642, 0.625, 0.626, 0.642, 0.643]
        auc = [0.745, 0.789, 0.789, 0.791, 0.787, 0.788, 0.800, 0.803]
        mce = [0.509, 0.492, 0.493, 0.494, 0.492, 0.494, 0.491, 0.491]
    elif method == "ig" and dataset == "squad_adversarial":
        acc = [0.655, 0.653, 0.665, 0.724, 0.666, 0.730, 0.687, 0.746]
        auc = [0.779, 0.779, 0.791, 0.845, 0.792, 0.847, 0.810, 0.858]
        mce = [0.476, 0.475, 0.474, 0.464, 0.474, 0.462, 0.475, 0.464]
    elif method == "ig" and dataset == "trivia_qa":
        acc = [0.729, 0.710, 0.705, 0.706, 0.702, 0.702, 0.706, 0.708]
        auc = [0.722, 0.724, 0.760, 0.763, 0.752, 0.756, 0.749, 0.753]
        mce = [0.554, 0.535, 0.508, 0.510, 0.511, 0.511, 0.521, 0.524]
    elif method == "ig" and dataset == "hotpot_qa":
        acc = [0.629, 0.636, 0.641, 0.644, 0.620, 0.619, 0.642, 0.643]
        auc = [0.746, 0.784, 0.789, 0.791, 0.785, 0.783, 0.799, 0.804]
        mce = [0.507, 0.493, 0.493, 0.494, 0.493, 0.495, 0.492, 0.492]


    if method == "conf" and dataset == "natural_questions":
        acc = [0.686, 0.71, 0.727, 0.72, 0.706]
        auc = [0.701, 0.78, 0.805, 0.813, 0.80]
        mce = [0.531, 0.516, 0.49, 0.476, 0.494]
    elif method == "conf" and dataset == "news_qa":
        acc = [0.68, 0.678, 0.699, 0.703, 0.687]
        auc = [0.713, 0.744, 0.751, 0.751, 0.767]
        mce = [0.535, 0.536, 0.530, 0.534, 0.527]
    elif method == "conf" and dataset == "bioasq":
        acc = [0.633, 0.67, 0.668, 0.651, 0.675]
        auc = [0.721, 0.754, 0.795, 0.802, 0.77]
        mce = [0.499, 0.507, 0.477, 0.465, 0.494]
    elif method == "shap" and dataset == "natural_questions":
        acc = [0.769, 0.747, 0.738, 0.738, 0.73, 0.731, 0.739, 0.741]
        auc = [0.779, 0.80, 0.803, 0.805, 0.81, 0.812, 0.814, 0.815]
        mce = [0.520, 0.491, 0.484, 0.486, 0.478, 0.480, 0.493, 0.494]
    elif method == "shap" and dataset == "news_qa":
        acc = [0.709, 0.703, 0.70, 0.70, 0.698, 0.698, 0.711, 0.712]
        auc = [0.752, 0.763, 0.753, 0.753, 0.74, 0.74, 0.774, 0.775]
        mce = [0.517, 0.515, 0.517, 0.516, 0.518, 0.517, 0.514, 0.514]
    elif method == "shap" and dataset == "bioasq":
        acc = [0.699, 0.713, 0.721, 0.72, 0.71, 0.697, 0.732, 0.721]
        auc = [0.768, 0.784, 0.822, 0.822, 0.824, 0.823, 0.81, 0.81]
        mce = [0.504, 0.496, 0.493, 0.493, 0.485, 0.485, 0.497, 0.497]
    elif method == "sc_attn" and dataset == "natural_questions":
        acc = [0.766, 0.741, 0.74, 0.741, 0.727, 0.729, 0.745, 0.744]
        auc = [0.777, 0.797, 0.804, 0.805, 0.809, 0.81, 0.813, 0.814]
        mce = [0.526, 0.496, 0.489, 0.49, 0.481, 0.482, 0.495, 0.496]
    elif method == "sc_attn" and dataset == "news_qa":
        acc = [0.720, 0.707, 0.704, 0.705, 0.703, 0.702, 0.713, 0.716]
        auc = [0.762, 0.77, 0.756, 0.758, 0.745, 0.748, 0.777, 0.781]
        mce = [0.538, 0.533, 0.528, 0.529, 0.526, 0.526, 0.527, 0.528]
    elif method == "sc_attn" and dataset == "bioasq":
        acc = [0.717, 0.723, 0.727, 0.724, 0.729, 0.72, 0.737, 0.733]
        auc = [0.781, 0.787, 0.825, 0.823, 0.834, 0.83, 0.810, 0.813]
        mce = [0.509, 0.502, 0.494, 0.495, 0.486, 0.487, 0.502, 0.501]
    elif method == "ig" and dataset == "natural_questions":
        acc = [0.767, 0.745, 0.74, 0.74, 0.73, 0.730, 0.742, 0.743]
        auc = [0.777, 0.797, 0.804, 0.805, 0.807, 0.808, 0.813, 0.814]
        mce = [0.526, 0.497, 0.489, 0.490, 0.482, 0.483, 0.496, 0.497]
    elif method == "ig" and dataset == "news_qa":
        acc = [0.720, 0.706, 0.706, 0.705, 0.70, 0.702, 0.714, 0.717]
        auc = [0.762, 0.767, 0.758, 0.76, 0.743, 0.746, 0.779, 0.783]
        mce = [0.536, 0.528, 0.526, 0.527, 0.525, 0.524, 0.528, 0.529]
    elif method == "ig" and dataset == "bioasq":
        acc = [0.704, 0.713, 0.716, 0.72, 0.716, 0.703, 0.731, 0.73]
        auc = [0.77, 0.777, 0.819, 0.822, 0.824, 0.825, 0.807, 0.809]
        mce = [0.507, 0.502, 0.495, 0.495, 0.488, 0.487, 0.502, 0.501]
    return acc, auc, mce

def visualize_calibration(method="conf", dataset="squad_adversarial"):

    acc, auc, mce =load_data(method,dataset)
    mce_flip = [round(1-m, 3) for m in mce]

    if method == "conf":
        header = ["Base", "RAG", "LLaMA", "GPT-NeoxT", "Flan-UL2"]
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
        # marker_sym = ['D', 's', '*', '^', 'o']
        dashes = ["solid", "dashed", (0, (3,2,1,2)), "dotted", "dashdot"]

    else:
        header = ["Base", "RAG", "LLaMA", "LLaMA + F", "GPT-NeoxT", "GPT-NeoxT + F", "Flan-UL2", "Flan-UL2 + F"]
        colors = ['#1f77b4', '#ff7f0e',  '#d62728', '#8c564b', '#9467bd', '#e377c2','#2ca02c', '#008080']
        dashes = ["solid",
                  "dashed",
                  (0, (3,2,1,2)),  # small line dashdot
                  (0, (5, 1)),
                  "dotted",
                  (0, (3, 1, 1, 1)),  # longer dashed
                  "dashdot",
                  (0, (4, 3, 1, 2, 1, 3))  # double dot line
                  ]
        # marker_sym = ['D', 's', '*', 'P', '^', 'X', 'o', 'H']

    data = pd.DataFrame({'Accuracy': acc, 'AUC': auc, '1-MCE': mce_flip})
    data = data.transpose()
    data.columns = header
    # print(data.head())

    figsize = (3.5, 2.5)  # Adjust the width and height values as desired
    plt.figure(figsize=figsize)

    plt.ylim(0.4, 0.90)
    # plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    ax = sns.lineplot(data=data, marker='o', palette=colors)
    # Set marker patterns for each model
    # Set marker patterns for each line
    from itertools import cycle

    marker_labels = header
    # markers = cycle(marker_sym)
    # for line in ax.lines:
    #     line.set_marker(next(markers))

    # Set line dash styles
    line_styles = cycle(dashes)
    color_palette = cycle(colors)
    for line in ax.lines:
        line.set_linestyle(next(line_styles))
        line.set_color(next(color_palette))

    # Hide the legend box
    legend = plt.legend()
    legend.remove()

    # legend_handles = []
    # for i, label in enumerate(marker_labels):
    #     color = ax.lines[i].get_color()
    #     handle, = plt.plot([], [],# marker=marker_sym[i],
    #                        color='black', label=marker_labels[i], #markersize=4,
    #                        # linestyle='None',
    #                        markeredgecolor='None', markerfacecolor=color)
    #     legend_handles.append(handle)
    #
    # # Show the legend
    # legend = plt.legend(handles=legend_handles, loc='lower left', prop={'size': 8})
    # for text in legend.get_texts():
    #     text.set_fontsize(8)  # Set the desired font size
    # plt.setp(legend.get_frame(), alpha=0.2)  # Optional: Set the background opacity
    # legend.get_frame().set_edgecolor('black')
    # plt.setp(legend.get_frame(), width=0.2)  # Set the width of the legend box

    # Set tick color and text color to black
    plt.tick_params(colors='black')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')

    # Set the font size of x-tick labels and y-tick labels


    # plt.xlabel("metrics", fontsize=12, color="r")
    # plt.ylabel("scores", fontsize=12, color="r")
    # plt.show()

    plt.tight_layout()
    # Hide the x-tick labels
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Show both x and y gridlines
    plt.grid(True, which='both')
    # plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_tick_params(labelbottom=False)

    plt.savefig(f"{method}_{dataset}_1.pdf")


def plot_bar_chart(methods: List, datasets: List, subplot_titles: List, show_legend: bool, save_path: str):
    """
    plot the calibration results for models
    """

    metrics = ["Accuracy", "AUC", "1-MCE"]
    complete_data: List = []
    # Define consistent colors for models
    # model_colors = {
    #     "Base": "#1f77b4",
    #     "RAG": "#ff7f0e",
    #     "LLaMA": "#2ca02c",
    #     "LLaMA + F<sub>d": "#8c564b",
    #     "GPT-NeoxT": "#d62728",
    #     "GPT-NeoxT + F<sub>d": "#e377c2",
    #     "Flan-UL2": "#9467bd",
    #     "Flan-UL2 + F<sub>d": "#17becf"
    # }

    # model_colors = {
    #     "Base": 'rgb(251,180,174)',
    #     "RAG": 'rgb(179,205,227)',
    #     "LLaMA": 'rgb(204,235,197)',
    #     "LLaMA + F<sub>d": 'rgb(222,203,228)',
    #     "GPT-NeoxT": 'rgb(254,217,166)',
    #     "GPT-NeoxT + F<sub>d": 'rgb(255,255,204)',
    #     "Flan-UL2": 'rgb(229,216,189)',
    #     "Flan-UL2 + F<sub>d": 'rgb(253,218,236)'
    # }

    # model_colors = {
    #     "Base": 'rgb(102, 197, 204)',
    #     "RAG":  'rgb(246, 207, 113)',
    #     "LLaMA":  'rgb(248, 156, 116)',
    #     "LLaMA + F<sub>d": 'rgb(220, 176, 242)',
    #     "GPT-NeoxT":  'rgb(135, 197, 95)',
    #     "GPT-NeoxT + F<sub>d":  'rgb(158, 185, 243)',
    #     "Flan-UL2":  'rgb(254, 136, 177)',
    #     "Flan-UL2 + F<sub>d":  'rgb(201, 219, 116)'
    # }
    #nein
    # model_colors = {
    #     "Base": 'rgb(102,194,165)',
    #     "RAG": 'rgb(252,141,98)',
    #     "LLaMA":  'rgb(141,160,203)',
    #     "LLaMA + F<sub>d":  'rgb(231,138,195)',
    #     "GPT-NeoxT": 'rgb(166,216,84)',
    #     "GPT-NeoxT + F<sub>d": 'rgb(255,217,47)',
    #     "Flan-UL2": 'rgb(229,196,148)',
    #     "Flan-UL2 + F<sub>d": 'rgb(179,179,179)'
    # }

    # model_colors = {
    #     "Base": 'rgb(141,211,199)',
    #     "RAG": 'rgb(255,255,179)',
    #     "LLaMA":  'rgb(190,186,218)',
    #     "LLaMA + F<sub>d": 'rgb(251,128,114)',
    #     "GPT-NeoxT":   'rgb(128,177,211)',
    #     "GPT-NeoxT + F<sub>d":  'rgb(253,180,98)',
    #     "Flan-UL2": 'rgb(179,222,105)',
    #     "Flan-UL2 + F<sub>d":  'rgb(252,205,229)'
    # }

    model_colors = {
        "Base": "#636EFA",
        "RAG": "#EF553B",
        "LLaMA": "#00CC96",
        "LLaMA + F<sub>d": "#B6E880",
        "GPT-NeoxT": "#FFA15A",
        "GPT-NeoxT + F<sub>d": "#FECB52",
        "Flan-UL2": "#FF6692",
        "Flan-UL2 + F<sub>d": "#FF97FF"
    }

    for i in range(len(methods)):
        data: Dict = {}
        data["Metric"] = metrics
        if methods[i] == "conf":
            models = ["Base", "RAG", "LLaMA", "GPT-NeoxT", "Flan-UL2"]
        else:
            models = ["Base", "RAG", "LLaMA", "LLaMA + F<sub>d", "GPT-NeoxT", "GPT-NeoxT + F<sub>d", "Flan-UL2", "Flan-UL2 + F<sub>d"]
        acc, auc, mce = load_data(methods[i], datasets[i])
        mce_flip = [round(1 - m, 3) for m in mce]
        df = pd.DataFrame({'Accuracy': acc, 'AUC': auc, '1-MCE': mce_flip})
        for idx in df.index:
            data[models[idx]] = df.loc[idx].tolist()
        # print(data)
        complete_data.append(data)

    # Create subplots with n row and m columns
    num_cols = 3
    num_rows = len(methods)//num_cols
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        shared_yaxes=False,
        subplot_titles=  subplot_titles, #  ["NQ", "News QA", "BioASQ"]
        horizontal_spacing=0.05,
        vertical_spacing=0.075
    )

    for i, _data in enumerate(complete_data):
        if show_legend:
            if i+1 in [1, 4]:
                showlegend = True
            else:
                showlegend = False
        else:
            showlegend = False
        subplot = go.Figure()
        if methods[i] == "conf":
            models = ["Base", "RAG", "LLaMA", "GPT-NeoxT", "Flan-UL2"]
            legend_models = models
        else:
            models = ["Base", "RAG", "LLaMA + F<sub>d", "GPT-NeoxT + F<sub>d", "Flan-UL2 + F<sub>d"]
            legend_models = [ "LLaMA + F<sub>d", "GPT-NeoxT + F<sub>d", "Flan-UL2 + F<sub>d"]
        for j, model in enumerate(models):
            subplot.add_trace(
                go.Bar(
                    x=_data["Metric"],
                    y=_data[model],
                    showlegend=showlegend and model in legend_models,
                    name=model,
                    marker_color=model_colors[model],
                )
            )
        # subplot.update_traces(marker=dict(line=dict(color='black', width=1)))  # Adjust color and width as needed
        # Add subplot to the main figure
        for trace in subplot.data:
            row = i // 3 + 1
            col = (i + 1) % 3 if (i + 1) % 3 != 0 else 3
            fig.add_trace(trace, row=row, col=col)

    # Apply layout to each subplot
    fig.update_layout(
        font=dict(family='Times New Roman', size=12, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis2=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis2=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis3=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis3=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                   linewidth=1.5,
                   linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis4=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis4=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis5=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis5=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis6=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis6=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis7=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis7=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis8=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis8=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis9=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis9=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis10=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis10=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis11=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis11=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        xaxis12=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black'),  # Optional: Move x-axis ticks outside
        yaxis12=dict(title_font=dict(size=16, color='black'), ticks="outside", mirror=True, showline=True,
                    linewidth=1.5,
                    linecolor='black', range=[0, 1]),  # Optional: Move y-axis ticks outside
        yaxis_title="CONF<br>Scores",
        yaxis4_title="SHAP<br>Scores",
        yaxis7_title="SC. ATTN.<br>Scores",
        yaxis10_title="IG<br>Scores",
    )
    # Move legend above subplot
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=16)))
    additional_y_ticks = [0, 0.25, 0.5, 0.75, 1]
    fig.update_yaxes(tickvals=additional_y_ticks)
    # fig.show()
    fig.update_layout(width=1200, height=700, template="ggplot2", margin=dict(t=10,),)
    pio.write_image(fig, save_path)


if __name__ == '__main__':
    # visualize_calibration(method="sc_attn", dataset="hotpot")
    # save_results_conf("./src/calibration/visualize/data/calib_results_exp.csv")
    plot_bar_chart(
        methods=["conf"]*3+["shap"]*3+["sc_attn"]*3+["ig"]*3,
        # datasets=["natural_questions", "news_qa", "bioasq"]*4,
        # subplot_titles=["NQ", "News QA", "BioASQ"],
        datasets=["squad_adversarial", "trivia_qa", "hotpot_qa"] * 4,
        subplot_titles =  ['SQuAD Adversarial', 'Trivia QA', 'Hotpot QA'],
        show_legend=True,
        save_path='calibration_plots_3.pdf',
    )
    # print(px.colors.qualitative.Plotly)
