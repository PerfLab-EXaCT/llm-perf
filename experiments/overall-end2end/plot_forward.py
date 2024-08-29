import matplotlib.pyplot as plt

data = [
    # A100_80GB GPT2 512
    {"task": "a100-1.3b-512-4-torch_full",      "time": 15.9010200881958},
    {"task": "a100-1.3b-512-4-torch_lora",      "time": 19.67398910522461},
    {"task": "a100-1.3b-512-4-exposer_lora",    "time": 27.651625061035155},
    {"task": "a100-1.3b-512-4-torch_adapter",   "time": 17.586442337036132},
    {"task": "a100-1.3b-512-4-exposer_adapter", "time": 26.87940601348877},
    {"task": "a100-1.3b-512-4-torch_bitfit",     "time": 15.85426429748535},
    {"task": "a100-1.3b-512-4-exposer_bitfit",  "time": 24.760115394592287},
    # A100_80GB GPT2 1024
    {"task": "a100-1.3b-1024-4-torch_full",      "time": 35.230371704101564}, 
    {"task": "a100-1.3b-1024-4-torch_lora",      "time": 39.0041805267334}, 
    {"task": "a100-1.3b-1024-4-exposer_lora",    "time": 36.94284812927246}, 
    {"task": "a100-1.3b-1024-4-torch_adapter",   "time": 36.34348045349121}, 
    {"task": "a100-1.3b-1024-4-exposer_adapter", "time": 35.23997665405273}, 
    {"task": "a100-1.3b-1024-4-torch_bitfit",     "time": 35.171655807495114}, 
    {"task": "a100-1.3b-1024-4-exposer_bitfit",  "time": 33.194393692016604}, 
    # A100_80GB GPT2-XL 512
    {"task": "a100-2.7b-512-4-torch_full",      "time": 113.308037109375},
    {"task": "a100-2.7b-512-4-torch_lora",      "time": 131.96740692138673},
    {"task": "a100-2.7b-512-4-exposer_lora",    "time": 141.42046173095704},
    {"task": "a100-2.7b-512-4-torch_adapter",   "time": 117.66259719848632}, 
    {"task": "a100-2.7b-512-4-exposer_adapter", "time": 133.74453796386717}, 
    {"task": "a100-2.7b-512-4-torch_bitfit",     "time": 113.1192318725586}, 
    {"task": "a100-2.7b-512-4-exposer_bitfit",  "time": 123.51498245239257}, 
    # A100_80GB GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-torch_full",      "time": 0},
    {"task": "a100-2.7b-1024-4-torch_lora",      "time": 291.2561950683594},
    {"task": "a100-2.7b-1024-4-exposer_lora",    "time": 209.1548681640625},
    {"task": "a100-2.7b-1024-4-torch_adapter",   "time": 268.59085754394533},
    {"task": "a100-2.7b-1024-4-exposer_adapter", "time": 190.80222778320314},
    {"task": "a100-2.7b-1024-4-torch_bitfit",     "time": 262.18094665527343},
    {"task": "a100-2.7b-1024-4-exposer_bitfit",  "time": 183.07086334228515},
]

tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 1.4
gap_width = 0.2
fontString = '21'
barOffset = 5

plt.figure(figsize=(15, 10))
plt.title("Batch Runtime for One Forward Pass", fontsize = 36)
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#E4080A']

# A100 GPT2 512
plt.bar(0, times[0], color=colors[0], width=default_width, edgecolor='black', label = 'Full Parameter')

plt.bar(2 - 0.1 , times[1], color=colors[1], width=default_width, edgecolor='black', label = 'LoRA')
plt.text(2 - 0.5, times[1] + barOffset, '0.81x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[1])

plt.bar(4 - 0.1, times[3], color=colors[2], width=default_width, edgecolor='black', label = 'Adapter')
plt.text(4 - 0.5, times[3] + barOffset, '0.9x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[3])

plt.bar(6 - 0.1, times[5], color=colors[3], width=default_width, edgecolor='black', label = 'BitFit')
plt.text(6 - 0.5, times[5] + barOffset, '1x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[5])


# A100 GPT2 1024
start2 = 9.3
plt.bar(start2, times[7], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start2 + 2 - 0.1, times[8], color=colors[1], width=default_width, edgecolor='black')
plt.text(start2 + 2 - 0.5, times[8] + barOffset, '0.9x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[8])

plt.bar(start2 + 4 - 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.text(start2 + 4 - 0.5, times[10] + barOffset, '0.97x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[10])

plt.bar(start2 + 6 - 0.1, times[12], color=colors[3], width=default_width, edgecolor='black')
plt.text(start2 + 6 - 0.5, times[12] + barOffset, '1x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[12])


# A100 GPT2-XL 512
start3 = 20.2
plt.bar(start3, times[14], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start3 + 2 - 0.1, times[15], color=colors[1], width=default_width, edgecolor='black')
plt.text(start3 + 2 - 0.5, times[15] + barOffset, '0.86x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[15])

plt.bar(start3 + 4 - 0.1, times[17], color=colors[2], width=default_width, edgecolor='black')
plt.text(start3 + 4 - 0.5, times[17] + barOffset, '0.96x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[17])

plt.bar(start3 + 6 - 0.1, times[19], color=colors[3], width=default_width, edgecolor='black')
plt.text(start3 + 6 - 0.5, times[19] + barOffset, '1x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
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
plt.text(6, 175, 'GPT-2', fontsize = 32)
plt.text(22, 175, 'GPT-2 XL', fontsize = 32)
plt.axvline(x=17.7, color = 'black', linestyle = '-.')

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(loc='upper left', fontsize=25, frameon=True)

#plt.legend(['_','Full Parameter', 'LoRA', 'Adapter', 'BitFit', 'Out of Memory'], loc='upper left', fontsize=10, frameon=True)
plt.savefig('graph_forward.pdf')
