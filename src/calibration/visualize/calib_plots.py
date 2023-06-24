import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

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


def visualize_calibration(method="conf", dataset="squad_adversarial"):

    acc, auc, mce = [], [], []
    if method=="conf" and dataset=="squad_adversarial":
        # conf squad
        acc = [0.642, 0.644, 0.656, 0.660, 0.666]
        auc = [0.685, 0.709, 0.716, 0.713, 0.749]
        mce = [0.474, 0.465, 0.467, 0.473, 0.454]
    elif method == "conf" and dataset == "trivia":
        # conf trivia
        acc = [0.601, 0.659, 0.669, 0.678, 0.665]
        auc = [0.583, 0.688, 0.748, 0.745, 0.732]
        mce = [0.539, 0.549, 0.507, 0.507, 0.520]
    elif method == "conf" and dataset == "hotpot":
        # conf hotpot
        acc = [0.611, 0.635, 0.643, 0.630, 0.651]
        auc = [0.740, 0.793, 0.800, 0.800, 0.816]
        mce = [0.502, 0.495, 0.487, 0.482, 0.488]
    elif method == "shap" and dataset == "squad_adversarial":
        # shap squad
        acc = [0.750, 0.736, 0.734, 0.776, 0.745, 0.784, 0.752, 0.789]
        auc = [0.843, 0.840, 0.840, 0.872, 0.847, 0.877, 0.854, 0.885]
        mce = [0.471, 0.469, 0.468, 0.459, 0.470, 0.459, 0.468, 0.460]
    elif method == "shap" and dataset == "trivia":
        # shap trivia
        acc = [0.720, 0.707, 0.707, 0.707, 0.703, 0.702, 0.707, 0.708]
        auc = [0.718, 0.723, 0.761, 0.763, 0.755, 0.755, 0.753, 0.753]
        mce = [0.545, 0.527, 0.505, 0.507, 0.508, 0.510, 0.515, 0.520]
    elif method == "shap" and dataset == "hotpot":
        # shap hotpot
        acc = [0.633, 0.646, 0.653, 0.654, 0.627, 0.628, 0.653, 0.653]
        auc = [0.755, 0.797, 0.800, 0.801, 0.796, 0.796, 0.811, 0.812]
        mce = [0.504, 0.490, 0.493, 0.494, 0.493, 0.493, 0.489, 0.490]
    elif method == "sc_attn" and dataset == "squad_adversarial":
        # sc attn squad
        acc = [0.657, 0.661, 0.654, 0.723, 0.670, 0.737, 0.672, 0.739]
        auc = [0.781, 0.794, 0.781, 0.848, 0.790, 0.855, 0.791, 0.860]
        mce = [0.476, 0.469, 0.475, 0.463, 0.475, 0.462, 0.479, 0.465]
    elif method == "sc_attn" and dataset == "trivia":
        # sc attn trivia
        acc = [0.735, 0.713, 0.705, 0.705, 0.699, 0.700, 0.707, 0.708]
        auc = [0.719, 0.722, 0.763, 0.765, 0.756, 0.757, 0.752, 0.753]
        mce = [0.558, 0.539, 0.510, 0.513, 0.511, 0.513, 0.520, 0.524]
    elif method == "sc_attn" and dataset == "hotpot":
        # sc attn hotpot
        acc = [0.631, 0.641, 0.640, 0.643, 0.625, 0.628, 0.642, 0.643]
        auc = [0.745, 0.789, 0.789, 0.792, 0.787, 0.790, 0.800, 0.802]
        mce = [0.509, 0.492, 0.493, 0.494, 0.492, 0.494, 0.491, 0.492]
    elif method == "ig" and dataset == "squad_adversarial":
        # ig squad
        acc = [0.655, 0.653, 0.665, 0.731, 0.666, 0.737, 0.687, 0.752]
        auc = [0.779, 0.779, 0.791, 0.848, 0.792, 0.854, 0.810, 0.862]
        mce = [0.476, 0.475, 0.474, 0.465, 0.474, 0.463, 0.475, 0.468]
    elif method == "ig" and dataset == "trivia":
        # ig trivia
        acc = [0.729, 0.710, 0.705, 0.706, 0.702, 0.704, 0.706, 0.707]
        auc = [0.722, 0.724, 0.760, 0.762, 0.752, 0.754, 0.749, 0.750]
        mce = [0.554, 0.535, 0.508, 0.512, 0.511, 0.513, 0.521, 0.526]
    elif method == "ig" and dataset == "hotpot":
        # ig hotpot
        acc = [0.629, 0.636, 0.641, 0.643, 0.620, 0.622, 0.642, 0.645]
        auc = [0.746, 0.784, 0.789, 0.791, 0.785, 0.787, 0.799, 0.803]
        mce = [0.507, 0.493, 0.493, 0.494, 0.493, 0.495, 0.492, 0.493]


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

    # plt.savefig(f"{method}_{dataset}_2.pdf")
    plt.show()
    # plt.savefig(f"{method}_{dataset}.svg", format="svg")


if __name__ == '__main__':
    visualize_calibration(method="shap", dataset="squad_adversarial")
