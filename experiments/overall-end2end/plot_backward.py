import matplotlib.pyplot as plt


data = [
    # A100_80GB GPT2 512
    {"task": "a100-1.3b-512-4-torch_full",      "time": 26.046463928222657},
    {"task": "a100-1.3b-512-4-torch_lora",      "time": 24.40003593444824},
    {"task": "a100-1.3b-512-4-exposer_lora",    "time": 24.070266876220703},
    {"task": "a100-1.3b-512-4-torch_adapter",   "time": 20.878643226623534},
    {"task": "a100-1.3b-512-4-exposer_adapter", "time": 19.8128231048584},
    {"task": "a100-1.3b-512-4-torch_bitfit",     "time": 20.620533714294435},
    {"task": "a100-1.3b-512-4-exposer_bitfit",  "time": 16.260280418395997},
    # A100_80GB GPT2 1024
    {"task": "a100-1.3b-1024-4-torch_full",      "time": 57.845801162719724}, 
    {"task": "a100-1.3b-1024-4-torch_lora",      "time": 53.41315071105957}, 
    {"task": "a100-1.3b-1024-4-exposer_lora",    "time": 34.588917846679685}, 
    {"task": "a100-1.3b-1024-4-torch_adapter",   "time": 47.39651596069336}, 
    {"task": "a100-1.3b-1024-4-exposer_adapter", "time": 30.37036521911621}, 
    {"task": "a100-1.3b-1024-4-torch_bitfit",     "time": 49.034915924072266}, 
    {"task": "a100-1.3b-1024-4-exposer_bitfit",  "time": 30.57264633178711}, 
    # A100_80GB GPT2-XL 512
    {"task": "a100-2.7b-512-4-torch_full",      "time": 188.6533837890625},
    {"task": "a100-2.7b-512-4-torch_lora",      "time": 181.3333203125},
    {"task": "a100-2.7b-512-4-exposer_lora",    "time": 141.42046173095704},
    {"task": "a100-2.7b-512-4-torch_adapter",   "time": 147.56294586181642}, 
    {"task": "a100-2.7b-512-4-exposer_adapter", "time": 133.74453796386717}, 
    {"task": "a100-2.7b-512-4-torch_bitfit",     "time": 146.9063983154297}, 
    {"task": "a100-2.7b-512-4-exposer_bitfit",  "time": 123.51498245239257}, 
    # A100_80GB GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-torch_full",      "time": 0},
    {"task": "a100-2.7b-1024-4-torch_lora",      "time": 412.2765509033203},
    {"task": "a100-2.7b-1024-4-exposer_lora",    "time": 209.1548681640625},
    {"task": "a100-2.7b-1024-4-torch_adapter",   "time": 364.44108825683594},
    {"task": "a100-2.7b-1024-4-exposer_adapter", "time": 190.80222778320314},
    {"task": "a100-2.7b-1024-4-torch_bitfit",     "time": 366.6363000488281},
    {"task": "a100-2.7b-1024-4-exposer_bitfit",  "time": 183.07086334228515},
]

tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 1.4
gap_width = 0.2
fontString = '21'
barOffset = 7

plt.figure(figsize=(15, 10))
plt.title("Batch Runtime for One Backward Pass", fontsize = 36)
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#E4080A']

# A100 GPT2 512
plt.bar(0, times[0], color=colors[0], width=default_width, edgecolor='black', label = 'Full Parameter')

plt.bar(2 - 0.1 , times[1], color=colors[1], width=default_width, edgecolor='black', label = 'LoRA')
plt.text(2 - 0.5, times[1] + barOffset, '1.07x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[1])

plt.bar(4 - 0.1, times[3], color=colors[2], width=default_width, edgecolor='black', label = 'Adapter')
plt.text(4 - 0.5, times[3] + barOffset, '1.25x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[3])

plt.bar(6 - 0.1, times[5], color=colors[3], width=default_width, edgecolor='black', label = 'BitFit')
plt.text(6 - 0.5, times[5] + barOffset, '1.26x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[5])


# A100 GPT2 1024
start2 = 9.3
plt.bar(start2, times[7], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start2 + 2 - 0.1, times[8], color=colors[1], width=default_width, edgecolor='black')
plt.text(start2 +  2 - 0.5, times[8] + barOffset, '1.08x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[8])

plt.bar(start2 + 4 - 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.text(start2 + 4 - 0.5, times[10] + barOffset, '1.22x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[10])

plt.bar(start2 + 6 - 0.1, times[12], color=colors[3], width=default_width, edgecolor='black')
plt.text(start2 + 6 - 0.5, times[12] + barOffset, '1.18x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[12])


# A100 GPT2-XL 512
start3 = 20.2
plt.bar(start3, times[14], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start3 + 2 - 0.1, times[15], color=colors[1], width=default_width, edgecolor='black')
plt.text(start3 + 2 - 0.5, times[15] + barOffset, '1.04x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[15])

plt.bar(start3 + 4 - 0.1, times[17], color=colors[2], width=default_width, edgecolor='black')
plt.text(start3 + 4 - 0.5, times[17] + barOffset, '1.28x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[17])

plt.bar(start3 + 6 - 0.1, times[19], color=colors[3], width=default_width, edgecolor='black')
plt.text(start3 + 6 - 0.5, times[19] + barOffset, '1.28x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[19])


# A100 GPT2-XL 1024
start4 = 29.5
plt.bar(start4, times[21], color=colors[4], width=default_width, edgecolor='black', label = 'Out of Memory')
plt.text(start4 - 0.1, 12, 'OOM', color = 'Red', rotation = 'vertical', fontsize = '28')

plt.bar(start4 + 2 - 0.1, times[22], color=colors[1], width=default_width, edgecolor='black')
print('speedup:', times[21] / times[22])

plt.bar(start4 + 4 - 0.1, times[24], color=colors[2], width=default_width, edgecolor='black')
print('speedup:', times[21] / times[24])

plt.bar(start4 + 6 - 0.1, times[26], color=colors[3], width=default_width, edgecolor='black')
print('speedup:', times[21] / times[26])

plt.ylabel('Time (ms)', fontsize=32)
plt.xlabel('Sequence Length', fontsize=32)
plt.xticks([2.9, 12.3, 23.1, 32.5],['512','1024','512','1024'], fontsize=32)
plt.yticks(fontsize=32)
plt.text(6, 245, 'GPT-2', fontsize = 32)
plt.text(22, 245, 'GPT-2 XL', fontsize = 32)
plt.axvline(x=17.7, color = 'black', linestyle = '-.')

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(loc='upper left', fontsize=25, frameon=True)

#plt.legend(['_','Full Parameter', 'LoRA', 'Adapter', 'BitFit', 'Out of Memory'], loc='upper left', fontsize=10, frameon=True)
plt.savefig('./graph_backward.pdf')
