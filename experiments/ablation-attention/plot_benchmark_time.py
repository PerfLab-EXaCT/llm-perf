import matplotlib.pyplot as plt  
from matplotlib.ticker import ScalarFormatter
import numpy as np 
  
data = [  
    {'layer': 0, 'dense_time': 0.00791, 'bigbird_time': 0.00491, 'longformer_time': 0.00504, 'shadowy_time': 0.00586, 'exposer_time': 0.00424, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25781, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.08950},  
    {'layer': 1, 'dense_time': 0.00783, 'bigbird_time': 0.00463, 'longformer_time': 0.00488, 'shadowy_time': 0.00584, 'exposer_time': 0.00434, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.12277},  
    {'layer': 2, 'dense_time': 0.00779, 'bigbird_time': 0.00460, 'longformer_time': 0.00486, 'shadowy_time': 0.00580, 'exposer_time': 0.00445, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25781, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.12863},  
    {'layer': 3, 'dense_time': 0.00779, 'bigbird_time': 0.00460, 'longformer_time': 0.00488, 'shadowy_time': 0.00575, 'exposer_time': 0.00440, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.26074, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.11505},  
    {'layer': 4, 'dense_time': 0.00780, 'bigbird_time': 0.00477, 'longformer_time': 0.00498, 'shadowy_time': 0.00588, 'exposer_time': 0.00464, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25488, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.10000},  
    {'layer': 5, 'dense_time': 0.00794, 'bigbird_time': 0.00472, 'longformer_time': 0.00504, 'shadowy_time': 0.00589, 'exposer_time': 0.00459, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09152},  
    {'layer': 6, 'dense_time': 0.00793, 'bigbird_time': 0.00474, 'longformer_time': 0.00500, 'shadowy_time': 0.00589, 'exposer_time': 0.00465, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25781, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09103},  
    {'layer': 7, 'dense_time': 0.00790, 'bigbird_time': 0.00477, 'longformer_time': 0.00500, 'shadowy_time': 0.00593, 'exposer_time': 0.00461, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25683, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09017},  
    {'layer': 8, 'dense_time': 0.00797, 'bigbird_time': 0.00478, 'longformer_time': 0.00505, 'shadowy_time': 0.00593, 'exposer_time': 0.00466, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09854},  
    {'layer': 9, 'dense_time': 0.00777, 'bigbird_time': 0.00462, 'longformer_time': 0.00495, 'shadowy_time': 0.00577, 'exposer_time': 0.00431, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25585, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09393},  
    {'layer': 10, 'dense_time': 0.00778, 'bigbird_time': 0.00462, 'longformer_time': 0.00492, 'shadowy_time': 0.00585, 'exposer_time': 0.00422, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25878, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09121},  
    {'layer': 11, 'dense_time': 0.00780, 'bigbird_time': 0.00465, 'longformer_time': 0.00489, 'shadowy_time': 0.00580, 'exposer_time': 0.00433, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.09231},
    {'layer': 12, 'dense_time': 0.00787, 'bigbird_time': 0.00463, 'longformer_time': 0.00491, 'shadowy_time': 0.00580, 'exposer_time': 0.00429, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25585, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.08245},  
    {'layer': 13, 'dense_time': 0.00780, 'bigbird_time': 0.00467, 'longformer_time': 0.00489, 'shadowy_time': 0.00579, 'exposer_time': 0.00438, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25878, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.07855},  
    {'layer': 14, 'dense_time': 0.00779, 'bigbird_time': 0.00465, 'longformer_time': 0.00493, 'shadowy_time': 0.00581, 'exposer_time': 0.00427, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25683, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.06921},  
    {'layer': 15, 'dense_time': 0.00782, 'bigbird_time': 0.00464, 'longformer_time': 0.00491, 'shadowy_time': 0.00586, 'exposer_time': 0.00426, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.26171, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.07983},  
    {'layer': 16, 'dense_time': 0.00780, 'bigbird_time': 0.00461, 'longformer_time': 0.00494, 'shadowy_time': 0.00584, 'exposer_time': 0.00439, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25683, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.07562},  
    {'layer': 17, 'dense_time': 0.00783, 'bigbird_time': 0.00462, 'longformer_time': 0.00492, 'shadowy_time': 0.00577, 'exposer_time': 0.00429, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25585, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.07702},  
    {'layer': 18, 'dense_time': 0.00781, 'bigbird_time': 0.00468, 'longformer_time': 0.00495, 'shadowy_time': 0.00580, 'exposer_time': 0.00423, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25781, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.06921},  
    {'layer': 19, 'dense_time': 0.00778, 'bigbird_time': 0.00461, 'longformer_time': 0.00493, 'shadowy_time': 0.00584, 'exposer_time': 0.00421, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25390, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.06524},  
    {'layer': 20, 'dense_time': 0.00779, 'bigbird_time': 0.00462, 'longformer_time': 0.00492, 'shadowy_time': 0.00582, 'exposer_time': 0.00423, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.05828},  
    {'layer': 21, 'dense_time': 0.00779, 'bigbird_time': 0.00462, 'longformer_time': 0.00493, 'shadowy_time': 0.00577, 'exposer_time': 0.00479, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25585, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.05834},  
    {'layer': 22, 'dense_time': 0.00799, 'bigbird_time': 0.00482, 'longformer_time': 0.00506, 'shadowy_time': 0.00592, 'exposer_time': 0.00466, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.03964},  
    {'layer': 23, 'dense_time': 0.00790, 'bigbird_time': 0.00484, 'longformer_time': 0.00498, 'shadowy_time': 0.00580, 'exposer_time': 0.00426, 'dense_sparsity': 1.0, 'bigbird_sparsity': 0.25976, 'longformer_sparsity': 0.33008, 'shadowy_sparsity': 0.51563, 'exposer_sparsity': 0.06744},
]

# Extracting layer labels and corresponding times for each layout  
layers = [d['layer'] for d in data]  
dense_times = [d['dense_time'] * 1000 for d in data]  
bigbird_times = [d['bigbird_time'] * 1000 for d in data]  
longformer_times = [d['longformer_time'] * 1000 for d in data]  
shadowy_times = [d['shadowy_time'] * 1000 for d in data]  
exposer_times = [d['exposer_time'] * 1000 for d in data]
dense_sparsity = [d['dense_sparsity'] for d in data]
bigbird_sparsity = [d['bigbird_sparsity'] for d in data]
longformer_sparsity = [d['longformer_sparsity'] for d in data]
shadowy_sparsity = [d['shadowy_sparsity'] for d in data]
exposer_sparsity = [d['exposer_sparsity'] for d in data]

# Calculate the speedup ratio for each layout
dense_speedup = []
bigbird_speedup = []
longformer_speedup = []
shadowy_speedup = []
exposer_speedup = []
for i in range(len(layers)):
    dense_speedup.append(dense_times[i] / dense_times[i])
    bigbird_speedup.append(dense_times[i] / bigbird_times[i])
    longformer_speedup.append(dense_times[i] / longformer_times[i])
    shadowy_speedup.append(dense_times[i] / shadowy_times[i])
    exposer_speedup.append(dense_times[i] / exposer_times[i])

# Number of groups  
num_layers = len(layers)  
  
# Setting up the bar width  
bar_width = 0.15 
  
# Setting the position of the bars on the x-axis  
r1 = np.arange(num_layers)  
r2 = [x + bar_width for x in r1]  
r3 = [x + bar_width for x in r2]  
r4 = [x + bar_width for x in r3]  
r5 = [x + bar_width for x in r4]

# Creating the figure
fig, ax1 = plt.subplots(figsize=(6.3, 1.5))
  
# Creating the bar plot
bar_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax1.bar(r1, dense_times, color=bar_colors[0], width=bar_width, edgecolor='black', label='Dense', linewidth=0.5)
ax1.bar(r2, shadowy_times, color=bar_colors[1], width=bar_width, edgecolor='black', label='Shadowy', linewidth=0.5)  
ax1.bar(r3, bigbird_times, color=bar_colors[2], width=bar_width, edgecolor='black', label='BigBird', linewidth=0.5)  
ax1.bar(r4, longformer_times, color=bar_colors[3], width=bar_width, edgecolor='black', label='Longformer', linewidth=0.5)  
ax1.bar(r5, exposer_times, color=bar_colors[4], width=bar_width, edgecolor='black', label='Exposer', linewidth=0.5)  

# Adding labels  
ax1.set_xlabel('Layer', fontsize=8)  
ax1.set_ylabel('Time (ms)', fontsize=8)
ax1.set_xticks([r + bar_width for r in range(num_layers)], layers)  
ax1.set_xticklabels([d['layer'] for d in data], fontsize=8)

# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.yticks(fontsize=8)

# Creating legend & title for the bar plot  
# ax1.legend(bbox_to_anchor=(-0.1, 1.12, 0., .202), loc='lower left', ncol=5, mode="expand", borderaxespad=0., frameon=False, fontsize=10)

# Creating a secondary y-axis for the sparsity ratio line plot 
# line_colors = ['#958EA2', '#582156', '#385E88', '#AA2070', '#EC008C'] 
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
ax2 = ax1.twinx()
# ax2.plot(r1, dense_speedup, color=line_colors[0], marker='o', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r2, shadowy_speedup, color=line_colors[1], marker='^', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r3, bigbird_speedup, color=line_colors[2], marker='s', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r4, longformer_speedup, color=line_colors[3], marker='d', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(r5, exposer_speedup, color=line_colors[4], marker='x', markersize=3, linestyle='--', linewidth=0.5, markeredgecolor='black', markeredgewidth=0.5)

# Adding labels for the secondary y-axis
ax2.set_ylabel('Speedup', fontsize=8)
plt.yticks(fontsize=8)

# Saving the figure
plt.tight_layout()
plt.savefig('./experiments/ablation/attention/exp-ablation-attn-time.pdf')
