import matplotlib.pyplot as plt    
import numpy as np    
    
data = [    
    # {"Seq_len": 128,  "Full Parameter": 0.0, "LoRA": 0.0, "Long Exposure+LoRA": 0.0, "Long Exposure+LoRA (optimal)": 0.0},    
    # {"Seq_len": 256,  "Full Parameter": 0.0, "LoRA": 0.0, "Long Exposure+LoRA": 0.0, "Long Exposure+LoRA (optimal)": 0.0},    
    # {"Seq_len": 512,  "Full Parameter": 0.0, "LoRA": 0.0, "Long Exposure+LoRA": 0.0, "Long Exposure+LoRA (optimal)": 0.0},    
    # {"Seq_len": 1024, "Full Parameter": 0.0, "LoRA": 0.0, "Long Exposure+LoRA": 0.0, "Long Exposure+LoRA (optimal)": 0.0},    
    # {"Seq_len": 2048, "Full Parameter": 0.0, "LoRA": 0.0, "Long Exposure+LoRA": 0.0, "Long Exposure+LoRA (optimal)": 0.0},    
]    

colors = ['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#171A39']    
    
# Extracting data    
seq_lens = [d["Seq_len"] for d in data]    
full_param = [d["Full Parameter"] for d in data]    
lora = [d["LoRA"] for d in data]    
long_exposure_lora = [d["Long Exposure+LoRA"] for d in data]    
long_exposure_lora_optimal = [d["Long Exposure+LoRA (optimal)"] for d in data]    
    
# Number of groups and bar width    
num_groups = len(seq_lens)    
bar_width = 0.18
    
# Setting the position of the bars on the x-axis    
r1 = np.arange(num_groups)    
r2 = [x + bar_width for x in r1]    
r3 = [x + bar_width for x in r2]    
r4 = [x + bar_width for x in r3]    
    
# Create the figure and the bar plot    
plt.figure(figsize=(4, 3))    
    
plt.bar(r1, full_param, color=colors[0], width=bar_width, label='Full Parameter', edgecolor='black')   
plt.bar(r2, lora, color=colors[1], width=bar_width, label='LoRA', edgecolor='black')    
plt.bar(r3, long_exposure_lora, color=colors[2], width=bar_width, label='Long Exposure', edgecolor='black')    
plt.bar(r4, long_exposure_lora_optimal, color=colors[3], width=bar_width, label='Long Exposure(optimal)', edgecolor='black')    
    
# Adding labels    
# plt.xlabel('Sequence Length')    
plt.ylabel('GPU Memory (GB)', fontsize=12)   
plt.yticks(fontsize=12)
plt.xticks([r + bar_width for r in range(num_groups)], seq_lens, fontsize=12)
    
# Creating legend & title for the bar plot    
# plt.legend(bbox_to_anchor=(-0.12, 1.2, 1.15, .202), loc='lower left', ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize=12)  

# Adding grid
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Saving the figure  
plt.tight_layout()  
plt.savefig('./experiments/overall-memory/exp-memory-1.3b.pdf') 
