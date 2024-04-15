import matplotlib.pyplot as plt  
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
# dense_sparsity = [1.0 - d['dense_sparsity'] for d in data]
bigbird_sparsity = [1.0 - d['bigbird_sparsity'] for d in data]
longformer_sparsity = [1.0 - d['longformer_sparsity'] for d in data]
shadowy_sparsity = [1.0 - d['shadowy_sparsity'] for d in data]
exposer_sparsity = [1.0 - d['exposer_sparsity'] for d in data]

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
plt.figure(figsize=(4, 3))
  
# Creating the bar plot
line_colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']
# plt.plot(r1, dense_sparsity, color=line_colors[0], marker='o', label='Dense', markersize=3, linewidth=1)
plt.plot(r2, shadowy_sparsity, color=line_colors[0], marker='^', label='Shadowy', markersize=3, linewidth=1)
plt.plot(r3, bigbird_sparsity, color=line_colors[1], marker='s', label='BigBird', markersize=3, linewidth=1)
plt.plot(r4, longformer_sparsity, color=line_colors[2], marker='d', label='Longformer', markersize=3, linewidth=1)
plt.plot(r5, exposer_sparsity, color=line_colors[4], marker='x', label='Long Exposure', markersize=3, linewidth=1)
  
# Adding labels  
plt.xlabel('Layer') 
plt.ylabel('Sparsity Ratio')
plt.xticks([r + bar_width for r in range(num_layers)], layers, fontsize=10, rotation=90)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=10)

# Creating legend & title for the bar plot  
# plt.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 1.02), loc='lower left', ncol=5, mode="expand", borderaxespad=0., frameon=False, fontsize=8)
plt.legend(loc='lower right', fontsize=10, frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)

# Saving the figure (optional)  
plt.tight_layout()
plt.savefig('./experiments/ablation/attention/exp-ablation-attn-sparsity.pdf')
