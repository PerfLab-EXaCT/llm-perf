import matplotlib.pyplot as plt


data = [
    # A100 GPT2 512
    {"task": "a100-1.3b-512-4-full parameter",          "time": 35.74724609375},
    {"task": "a100-1.3b-512-4-lora",                    "time": 31.566294975280762},
    {"task": "a100-1.3b-512-4-long exposure + lora",    "time": 45.8922802734375},
    {"task": "a100-1.3b-512-4-adapter",                 "time": 30.295531463623046},
    {"task": "a100-1.3b-512-4-long exposure + adapter", "time": 45.30593803405762},
    {"task": "a100-1.3b-512-4-bitfit",                  "time": 23.78166248321533},
    {"task": "a100-1.3b-512-4-long exposure + bitfit",  "time": 39.44749084472656},
    # A100 GPT2 1024
    {"task": "a100-1.3b-1024-4-full parameter",          "time": 40.288973083496096},
    {"task": "a100-1.3b-1024-4-lora",                    "time": 36.91393020629883},
    {"task": "a100-1.3b-1024-4-long exposure + lora",    "time": 49.914736862182615},
    {"task": "a100-1.3b-1024-4-adapter",                 "time": 31.925637283325194},
    {"task": "a100-1.3b-1024-4-long exposure + adapter", "time": 46.824734649658204},
    {"task": "a100-1.3b-1024-4-bitfit",                  "time": 27.328634986877443},
    {"task": "a100-1.3b-1024-4-long exposure + bitfit",  "time": 40.9920516204834},
    # A100 GPT2-XL 512
    {"task": "a100-2.7b-512-4-full parameter",          "time": 216.3977835083008},
    {"task": "a100-2.7b-512-4-lora",                    "time": 130.34934219360352},
    {"task": "a100-2.7b-512-4-long exposure + lora",    "time": 187.4975750732422},
    {"task": "a100-2.7b-512-4-adapter",                 "time": 128.24336380004883},
    {"task": "a100-2.7b-512-4-long exposure + adapter", "time": 188.17663024902345},
    {"task": "a100-2.7b-512-4-bitfit",                  "time": 101.62325515747071},
    {"task": "a100-2.7b-512-4-long exposure + bitfit",  "time": 160.20690887451173},
    # A100 GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-full parameter",          "time": 337.4976599121094},
    {"task": "a100-2.7b-1024-4-lora",                    "time": 233.9647900390625},
    {"task": "a100-2.7b-1024-4-long exposure + lora",    "time": 209.88395446777344},
    {"task": "a100-2.7b-1024-4-adapter",                 "time": 204.8331164550781},
    {"task": "a100-2.7b-1024-4-long exposure + adapter", "time": 201.438740234375},
    {"task": "a100-2.7b-1024-4-bitfit",                  "time": 197.30524169921875},
    {"task": "a100-2.7b-1024-4-long exposure + bitfit",  "time": 179.47330474853516},
]


tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 0.8
gap_width = 0.2

plt.figure(figsize=(10, 10))
plt.title("GPT-2 A100 Training runtime", fontsize = 'xx-large')
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3']

# A100 GPT2 512
plt.bar(1 - 0.1, times[0], color=colors[0], width=default_width, edgecolor='black')

plt.bar(2 + 0.1, times[1], color=colors[1], width=default_width, edgecolor='black')
plt.bar(3 - 0.1, times[2], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[1] / times[2])

plt.bar(4 + 0.1, times[3], color=colors[2], width=default_width, edgecolor='black')
plt.bar(5 - 0.1, times[4], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[3] / times[4])

plt.bar(6 + 0.1, times[5], color=colors[3], width=default_width, edgecolor='black')
plt.bar(7 - 0.1, times[6], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[5] / times[6])

# A100 GPT2 1024
plt.bar(9 - 0.1, times[7], color=colors[0], width=default_width, edgecolor='black')

plt.bar(10 + 0.1, times[8], color=colors[1], width=default_width, edgecolor='black')
plt.bar(11 - 0.1, times[9], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[8] / times[9])

plt.bar(12 + 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.bar(13 - 0.1, times[11], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[10] / times[11])

plt.bar(14 + 0.1, times[12], color=colors[3], width=default_width, edgecolor='black')
plt.bar(15 - 0.1, times[13], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[12] / times[13])

# A100 GPT2-XL 512
plt.bar(17 - 0.1 + 1, times[14], color=colors[0], width=default_width, edgecolor='black')

plt.bar(18 + 0.1 + 1, times[15], color=colors[1], width=default_width, edgecolor='black')
plt.bar(19 - 0.1 + 1, times[16], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[15] / times[16])

plt.bar(20 + 0.1 + 1, times[17], color=colors[2], width=default_width, edgecolor='black')
plt.bar(21 - 0.1 + 1, times[18], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[17] / times[18])

plt.bar(22 + 0.1 + 1, times[19], color=colors[3], width=default_width, edgecolor='black')
plt.bar(23 - 0.1 + 1, times[20], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[19] / times[20])

# A100 GPT2-XL 1024
plt.bar(25 - 0.1 + 1, times[21], color=colors[0], width=default_width, edgecolor='black')

plt.bar(26 + 0.1 + 1, times[22], color=colors[1], width=default_width, edgecolor='black')
plt.bar(27 - 0.1 + 1, times[23], color=colors[1], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[22] / times[23])

plt.bar(28 + 0.1 + 1, times[24], color=colors[2], width=default_width, edgecolor='black')
plt.bar(29 - 0.1 + 1, times[25], color=colors[2], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[24] / times[25])

plt.bar(30 + 0.1 + 1, times[26], color=colors[3], width=default_width, edgecolor='black')
plt.bar(31 - 0.1 + 1, times[27], color=colors[3], width=default_width, edgecolor='black', hatch='////')
print('speedup:', times[26] / times[27])

plt.ylabel('Time (ms)', fontsize=16)
plt.xlabel('Sequence Length', fontsize=16)
plt.xticks([4 + 0.1, 12 + 0.1, 20 + 0.1 + 1, 28 + 0.1 + 1],['512','1024','512','1024'], fontsize=16)
plt.text(7, 250, 'GPT-2', fontsize = 15)
plt.text(20, 250, 'GPT-2 XL', fontsize = 15)
plt.axvline(x=16.3, color = 'black', linestyle = '-.')

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(['_','Full Parameter', 'LoRA', 'Long Exposure + LoRA', 'Adapter', 'Long Exposure + Adapter', 'BitFit', 'Long Exposure + BitFit'], loc='upper left', fontsize=10, frameon=True)

#plt.tight_layout()
#plt.savefig('./experiments/overall-end2end/exp-end2end-a100.pdf')
plt.savefig('./exp-end2end-a100.pdf')
