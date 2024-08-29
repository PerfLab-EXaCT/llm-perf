import matplotlib.pyplot as plt


data = [
    # A100_80GB GPT2 512
    {"task": "a100-1.3b-512-4-full parameter",          "time": 48.42280960083008},
    {"task": "a100-1.3b-512-4-lora",                    "time": 45.46433029174805},
    {"task": "a100-1.3b-512-4-long exposure + lora",    "time": 52.81757209777832},
    {"task": "a100-1.3b-512-4-adapter",                 "time": 38.611578674316405},
    {"task": "a100-1.3b-512-4-long exposure + adapter", "time": 48.49358848571777},
    {"task": "a100-1.3b-512-4-bitfit",                  "time": 36.544061431884764},
    {"task": "a100-1.3b-512-4-long exposure + bitfit",  "time": 42.969046936035156},
    # A100_80GB GPT2 1024
    {"task": "a100-1.3b-1024-4-full parameter",          "time": 99.53218551635742}, 
    {"task": "a100-1.3b-1024-4-lora",                    "time": 93.81531616210937}, 
    {"task": "a100-1.3b-1024-4-long exposure + lora",    "time": 72.9012629699707}, 
    {"task": "a100-1.3b-1024-4-adapter",                 "time": 83.89249069213867}, 
    {"task": "a100-1.3b-1024-4-long exposure + adapter", "time": 65.76101364135742}, 
    {"task": "a100-1.3b-1024-4-bitfit",                  "time": 84.27450347900391}, 
    {"task": "a100-1.3b-1024-4-long exposure + bitfit",  "time": 63.83601676940918}, 
    # A100_80GB GPT2-XL 512
    {"task": "a100-2.7b-512-4-full parameter",          "time": 381.8438250732422},
    {"task": "a100-2.7b-512-4-lora",                    "time": 317.50563720703127},
    {"task": "a100-2.7b-512-4-long exposure + lora",    "time": 290.83983642578124},
    {"task": "a100-2.7b-512-4-adapter",                 "time": 266.0228698730469}, 
    {"task": "a100-2.7b-512-4-long exposure + adapter", "time": 246.2791683959961}, 
    {"task": "a100-2.7b-512-4-bitfit",                  "time": 260.2553344726563}, 
    {"task": "a100-2.7b-512-4-long exposure + bitfit",  "time": 234.0199432373047}, 
    # A100_80GB GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-full parameter",          "time": 0},
    {"task": "a100-2.7b-1024-4-lora",                    "time": 707.47529296875},
    {"task": "a100-2.7b-1024-4-long exposure + lora",    "time": 472.952626953125},
    {"task": "a100-2.7b-1024-4-adapter",                 "time": 633.8401098632812},
    {"task": "a100-2.7b-1024-4-long exposure + adapter", "time": 406.17924743652344},
    {"task": "a100-2.7b-1024-4-bitfit",                  "time": 629.0448388671875},
    {"task": "a100-2.7b-1024-4-long exposure + bitfit",  "time": 397.0991522216797},
]


tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 1.4
gap_width = 0.2
fontString = '21'
barOffset = 10

plt.figure(figsize=(15, 10))
plt.title("Batch Runtime for One Training Loop", fontsize = 36)
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#E4080A']

# A100 GPT2 512
plt.bar(0, times[0], color=colors[0], width=default_width, edgecolor='black', label = 'Full Parameter')

plt.bar(2 - 0.1 , times[1], color=colors[1], width=default_width, edgecolor='black', label = 'LoRA')
plt.text(2 - 0.5, times[1] +barOffset, '1.07x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[1])

plt.bar(4 - 0.1, times[3], color=colors[2], width=default_width, edgecolor='black', label = 'Adapter')
plt.text(4 - 0.5, times[3] +barOffset, '1.25x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[3])

plt.bar(6 - 0.1, times[5], color=colors[3], width=default_width, edgecolor='black', label = 'BitFit')
plt.text(6 - 0.5, times[5] +barOffset, '1.33x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[5])

# A100 GPT2 1024
start2 = 9.3
plt.bar(start2, times[7], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start2 + 2 - 0.1, times[8], color=colors[1], width=default_width, edgecolor='black')
plt.text(start2 + 2 - 0.5, times[8] +barOffset, '1.06x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[8])

plt.bar(start2 + 4 - 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.text(start2 + 4 - 0.5, times[10] +barOffset, '1.19x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[10])

plt.bar(start2 + 6 - 0.1, times[12], color=colors[3], width=default_width, edgecolor='black')
plt.text(start2 + 6 - 0.5, times[12] +barOffset, '1.18x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[7] / times[12])

# A100 GPT2-XL 512
start3 = 20.2
plt.bar(start3, times[14], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start3 + 2 - 0.1, times[15], color=colors[1], width=default_width, edgecolor='black')
plt.text(start3 + 2 - 0.5, times[15] +barOffset, '1.20x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[15])

plt.bar(start3 + 4 - 0.1, times[17], color=colors[2], width=default_width, edgecolor='black')
plt.text(start3 + 4 - 0.5, times[17] +barOffset, '1.44x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[14] / times[17])

plt.bar(start3 + 6 - 0.1, times[19], color=colors[3], width=default_width, edgecolor='black')
plt.text(start3 + 6 - 0.5, times[19] +barOffset, '1.47x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
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
plt.text(6, 425, 'GPT-2', fontsize = 32)
plt.text(22, 425, 'GPT-2 XL', fontsize = 32)
plt.axvline(x=17.7, color = 'black', linestyle = '-.')

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(loc='upper left', fontsize=25, frameon=True)

#plt.legend(['_','Full Parameter', 'LoRA', 'Adapter', 'BitFit', 'Out of Memory'], loc='upper left', fontsize=10, frameon=True)
plt.savefig('graph_trainingloop.pdf')
