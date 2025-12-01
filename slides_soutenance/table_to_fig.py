datasets = {
    "room":    {"N": 8103,   "p": 16},
    "bean":    {"N": 10888,  "p": 16},
    "eeg":     {"N": 11984,  "p": 14},
    "avila":   {"N": 10430,  "p": 10},
    "magic":   {"N": 15216,  "p": 10},
    "htru":    {"N": 14318,  "p": 8},
    "occup.":  {"N": 8143,   "p": 5},
    "skin":    {"N": 196045, "p": 3},
    "fault":   {"N": 1552,   "p": 27},
    "segment": {"N": 1848,   "p": 18},
    "page":    {"N": 4378,   "p": 10},
    "bidding": {"N": 5056,   "p": 9},
    "raisin":  {"N": 720,    "p": 7},
    "rice":    {"N": 3048,   "p": 7},
    "wilt":    {"N": 4339,   "p": 5},
    "bank":    {"N": 1097,   "p": 4},
}

accuracies = {
    "room":    {"Opt": 0.992, "Greedy": 0.968, "DPDT_light": 0.991, "DPDT_full": 0.992, "TopB_light": 0.990, "TopB_full": 0.992, "DeepRL": 0.715},
    "bean":    {"Opt": 0.871, "Greedy": 0.777, "DPDT_light": 0.812, "DPDT_full": 0.853, "TopB_light": 0.804, "TopB_full": 0.841, "DeepRL": 0.182},
    "eeg":     {"Opt": 0.708, "Greedy": 0.666, "DPDT_light": 0.689, "DPDT_full": 0.706, "TopB_light": 0.684, "TopB_full": 0.699, "DeepRL": 0.549},
    "avila":   {"Opt": 0.585, "Greedy": 0.532, "DPDT_light": 0.574, "DPDT_full": 0.585, "TopB_light": 0.563, "TopB_full": 0.572, "DeepRL": 0.409},
    "magic":   {"Opt": 0.831, "Greedy": 0.801, "DPDT_light": 0.822, "DPDT_full": 0.828, "TopB_light": 0.807, "TopB_full": 0.816, "DeepRL": 0.581},
    "htru":    {"Opt": 0.981, "Greedy": 0.979, "DPDT_light": 0.979, "DPDT_full": 0.980, "TopB_light": 0.979, "TopB_full": 0.980, "DeepRL": 0.860},
    "occup.":  {"Opt": 0.994, "Greedy": 0.989, "DPDT_light": 0.991, "DPDT_full": 0.994, "TopB_light": 0.990, "TopB_full": 0.992, "DeepRL": 0.647},
    "skin":    {"Opt": 0.969, "Greedy": 0.966, "DPDT_light": 0.966, "DPDT_full": 0.966, "TopB_light": 0.966, "TopB_full": 0.966, "DeepRL": 0.612},
    "fault":   {"Opt": 0.682, "Greedy": 0.553, "DPDT_light": 0.672, "DPDT_full": 0.674, "TopB_light": 0.672, "TopB_full": 0.673, "DeepRL": 0.303},
    "segment": {"Opt": 0.887, "Greedy": 0.574, "DPDT_light": 0.812, "DPDT_full": 0.879, "TopB_light": 0.786, "TopB_full": 0.825, "DeepRL": 0.137},
    "page":    {"Opt": 0.971, "Greedy": 0.964, "DPDT_light": 0.970, "DPDT_full": 0.970, "TopB_light": 0.964, "TopB_full": 0.965, "DeepRL": 0.902},
    "bidding": {"Opt": 0.993, "Greedy": 0.981, "DPDT_light": 0.985, "DPDT_full": 0.993, "TopB_light": 0.985, "TopB_full": 0.993, "DeepRL": 0.810},
    "raisin":  {"Opt": 0.894, "Greedy": 0.869, "DPDT_light": 0.879, "DPDT_full": 0.886, "TopB_light": 0.875, "TopB_full": 0.883, "DeepRL": 0.509},
    "rice":    {"Opt": 0.938, "Greedy": 0.933, "DPDT_light": 0.934, "DPDT_full": 0.937, "TopB_light": 0.933, "TopB_full": 0.936, "DeepRL": 0.519},
    "wilt":    {"Opt": 0.996, "Greedy": 0.993, "DPDT_light": 0.994, "DPDT_full": 0.995, "TopB_light": 0.994, "TopB_full": 0.994, "DeepRL": 0.984},
    "bank":    {"Opt": 0.983, "Greedy": 0.933, "DPDT_light": 0.971, "DPDT_full": 0.980, "TopB_light": 0.951, "TopB_full": 0.974, "DeepRL": 0.496},
}

operations = {
    "room":    {"Opt": 1e6,      "Greedy": 15, "DPDT_light": 286, "DPDT_full": 16100, "TopB_light": 111, "TopB_full": 16100},
    "bean":    {"Opt": 5e6,      "Greedy": 15, "DPDT_light": 295, "DPDT_full": 25900, "TopB_light": 112, "TopB_full": 16800},
    "eeg":     {"Opt": 2e6,      "Greedy": 13, "DPDT_light": 289, "DPDT_full": 26000, "TopB_light": 95,  "TopB_full": 11000},
    "avila":   {"Opt": 3e7,      "Greedy": 9,  "DPDT_light": 268, "DPDT_full": 24700, "TopB_light": 60,  "TopB_full": 38900},
    "magic":   {"Opt": 6e6,      "Greedy": 15, "DPDT_light": 298, "DPDT_full": 28000, "TopB_light": 70,  "TopB_full": 4190},
    "htru":    {"Opt": 6e7,      "Greedy": 15, "DPDT_light": 295, "DPDT_full": 25300, "TopB_light": 55,  "TopB_full": 2180},
    "occup.":  {"Opt": 7e5,      "Greedy": 13, "DPDT_light": 280, "DPDT_full": 16300, "TopB_light": 33,  "TopB_full": 510},
    "skin":    {"Opt": 7e4,      "Greedy": 15, "DPDT_light": 301, "DPDT_full": 23300, "TopB_light": 20,  "TopB_full": 126},
    "fault":   {"Opt": 9e8,      "Greedy": 13, "DPDT_light": 295, "DPDT_full": 24200, "TopB_light": 111, "TopB_full": 16800},
    "segment": {"Opt": 2e6,      "Greedy": 7,  "DPDT_light": 220, "DPDT_full": 16300, "TopB_light": 68,  "TopB_full": 11400},
    "page":    {"Opt": 1e7,      "Greedy": 15, "DPDT_light": 298, "DPDT_full": 22400, "TopB_light": 701, "TopB_full": 4050},
    "bidding": {"Opt": 3e5,      "Greedy": 13, "DPDT_light": 256, "DPDT_full": 9360,  "TopB_light": 58,  "TopB_full": 2700},
    "raisin":  {"Opt": 4e6,      "Greedy": 15, "DPDT_light": 295, "DPDT_full": 20900, "TopB_light": 48,  "TopB_full": 1440},
    "rice":    {"Opt": 2e7,      "Greedy": 15, "DPDT_light": 298, "DPDT_full": 25500, "TopB_light": 49,  "TopB_full": 1470},
    "wilt":    {"Opt": 3e5,      "Greedy": 13, "DPDT_light": 274, "DPDT_full": 11300, "TopB_light": 33,  "TopB_full": 465},
    "bank":    {"Opt": 6e4,      "Greedy": 13, "DPDT_light": 271, "DPDT_full": 7990,  "TopB_light": 26,  "TopB_full": 256},
}

import matplotlib.pyplot as plt
import numpy as np
# Your algorithms
algos = ['DPDT_full', 'TopB_full']

# Dataset names (keys of the accuracies dict)
dataset_names = list(accuracies.keys())

# Compute normalized accuracies
data = {
    algo: [(accuracies[d][algo] -accuracies[d]['Greedy'] )/ (accuracies[d]['Opt'] - accuracies[d]['Greedy']) for d in dataset_names]
    for algo in algos
}

# Bar positions
x = np.arange(len(dataset_names))
width = 0.25  # width of each bar

plt.figure(figsize=(12, 6))
labels = {'DPDT_full': 'DPDT+CART', 'TopB_full': 'DPDT+Top B'}
colors = {'Greedy': 'brown', 'DPDT_full': 'tab:blue', 'DPDT_light':'darkturquoise', 'TopB_full': 'tab:red', 'TopB_light': 'limegreen'}
# Plot each algorithm as a shifted bar
for i, algo in enumerate(algos):
    plt.bar(x + i * width, data[algo], width, label=labels[algo], color=colors[algo])

# Axes & labels
plt.ylim(0.5, 1)
plt.xticks(x + width * 2, dataset_names, rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=18)
plt.ylabel("Normalized train accuracy",fontdict={'fontsize':18})
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('table_to_fig.pdf')
plt.clf()

data = {
    algo: [(operations[d][algo]-operations[d]['Greedy'] )/ (operations[d]['Opt'] - operations[d]['Greedy']) for d in dataset_names]
    for algo in algos
}

# Bar positions
x = np.arange(len(dataset_names))
width = 0.25  # width of each bar

plt.figure(figsize=(12, 6))
labels = {'DPDT_full': 'DPDT+CART', 'TopB_full': 'DPDT+Top B'}
colors = {'Greedy': 'brown', 'DPDT_full': 'tab:blue', 'DPDT_light':'darkturquoise', 'TopB_full': 'tab:red', 'TopB_light': 'limegreen'}
# Plot each algorithm as a shifted bar
for i, algo in enumerate(algos):
    plt.bar(x + i * width, data[algo], width, label=labels[algo], color=colors[algo])
plt.yscale('log')
# Axes & labels
plt.xticks(x + width * 2, dataset_names, rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=18)
plt.ylabel("Normalized #operations",fontdict={'fontsize':18})
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('table_to_fig_cplx.pdf')

plt.clf()
algos = ['DPDT_full', 'TopB_full']
labels = {'DPDT_full': 'DPDT+CART', 'TopB_full': 'DPDT+Top B'}
colors = {'DPDT_full': 'tab:blue', 'TopB_full': 'tab:red'}

dataset_names = list(accuracies.keys())
x = np.arange(len(dataset_names))
width = 0.35

# ---- Normalized accuracy ----
acc_data = {
    algo: [(accuracies[d][algo] - accuracies[d]['Greedy']) /
           (accuracies[d]['Opt'] - accuracies[d]['Greedy'])
           for d in dataset_names]
    for algo in algos
}

# ---- Normalized cost ----
cost_data = {
    algo: [(operations[d][algo] - operations[d]['Greedy']) /
           (operations[d]['Opt'] - operations[d]['Greedy'])
           for d in dataset_names]
    for algo in algos
}

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(13, 8), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}
)

# ---------------- TOP: Accuracy ----------------
for i, algo in enumerate(algos):
    ax_top.bar(x + i*width, acc_data[algo], width,
               label=labels[algo], color=colors[algo])

# ax_top.set_ylim(-0.1, 1.05)

ax_top.set_ylabel("Normalized accuracy", fontsize=18)
ax_top.legend(fontsize=22)
ax_top.grid(axis='y', linestyle='--', alpha=0.6)

# ---------------- BOTTOM: Cost (mirrored) ----------------
for i, algo in enumerate(algos):
    # negative values for mirror effect
    ax_bottom.bar(x + i*width, np.array(cost_data[algo]), width,
                  color=colors[algo])

ax_bottom.set_yscale('log')   # symmetric log handles negative values
ax_bottom.set_ylabel("Log normalized cost", fontsize=18)
ax_top.yaxis.set_tick_params(labelsize=17)
ax_bottom.grid(axis='y', linestyle='--', alpha=0.6)

ax_bottom.set_xticks(x + width/2)
ax_bottom.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig("mirror_accuracy_vs_cost.pdf")