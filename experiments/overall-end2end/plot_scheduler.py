import matplotlib.pyplot as plt


data = [
    # A100_80GB GPT2 512
    {"task": "a100-1.3b-512-4-torch_full",      "time": 0.0036864000372588634},
    {"task": "a100-1.3b-512-4-torch_lora",      "time": 0.021503999717533587},
    {"task": "a100-1.3b-512-4-torch_adapter",   "time": 0.003788800025358796},
    {"task": "a100-1.3b-512-4-torch_bitfit",     "time": 0.0037273600324988364},
    # A100_80GB GPT2 1024
    {"task": "a100-1.3b-1024-4-torch_full",      "time": 0.003624960044398904}, 
    {"task": "a100-1.3b-1024-4-torch_lora",      "time": 0.021852159835398198}, 
    {"task": "a100-1.3b-1024-4-torch_adapter",   "time": 0.0037683200277388094}, 
    {"task": "a100-1.3b-1024-4-torch_bitfit",     "time": 0.0037273600324988364}, 
    # A100_80GB GPT2-XL 512
    {"task": "a100-2.7b-512-4-torch_full",      "time": 0.0038092800229787825},
    {"task": "a100-2.7b-512-4-torch_lora",      "time": 0.02211839996278286},
    {"task": "a100-2.7b-512-4-torch_adapter",   "time": 0.0038092800229787825}, 
    {"task": "a100-2.7b-512-4-torch_bitfit",     "time": 0.0035840000491589308}, 
    # A100_80GB GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-torch_full",      "time": 0},
    {"task": "a100-2.7b-1024-4-torch_lora",      "time": 0.02166783984750509},
    {"task": "a100-2.7b-1024-4-torch_adapter",   "time": 0.003911680011078715},
    {"task": "a100-2.7b-1024-4-torch_bitfit",     "time": 0.003829760020598769},
]

tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 1.4
gap_width = 0.2
fontString = '21'
barOffset = 0.0008

plt.figure(figsize=(15, 10))
plt.title("Batch Runtime for One Scheduler Pass", fontsize = 36)
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#E4080A']

# A100 GPT2 512
plt.bar(0, times[0], color=colors[0], width=default_width, edgecolor='black', label = 'Full Parameter')

plt.bar(2 - 0.1 , times[1], color=colors[1], width=default_width, edgecolor='black', label = 'LoRA')
plt.text(2 - 0.5, times[1] + barOffset, '0.17x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[1])

plt.bar(4 - 0.1, times[2], color=colors[2], width=default_width, edgecolor='black', label = 'Adapter')
plt.text(4 - 0.5, times[2] + barOffset, '0.97x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[2])

plt.bar(6 - 0.1, times[3], color=colors[3], width=default_width, edgecolor='black', label = 'BitFit')
plt.text(6 - 0.5, times[3] + barOffset, '0.99x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[3])


# A100 GPT2 1024
start2 = 9.3
plt.bar(start2, times[4], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start2 + 2 - 0.1, times[5], color=colors[1], width=default_width, edgecolor='black')
plt.text(start2 +  2 - 0.5, times[5] + barOffset, '0.17x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[5])

plt.bar(start2 + 4 - 0.1, times[6], color=colors[2], width=default_width, edgecolor='black')
plt.text(start2 + 4 - 0.5, times[6] + barOffset, '0.96x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[6])

plt.bar(start2 + 6 - 0.1, times[7], color=colors[3], width=default_width, edgecolor='black')
plt.text(start2 + 6 - 0.5, times[7] + barOffset, '0.97x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[7])


# A100 GPT2-XL 512
start3 = 20.2
plt.bar(start3, times[8], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start3 + 2 - 0.1, times[9], color=colors[1], width=default_width, edgecolor='black')
plt.text(start3 + 2 - 0.5, times[9] + barOffset, '0.17x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[9])

plt.bar(start3 + 4 - 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.text(start3 + 4 - 0.5, times[10] + barOffset, '1x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[10])

plt.bar(start3 + 6 - 0.1, times[11], color=colors[3], width=default_width, edgecolor='black')
plt.text(start3 + 6 - 0.5, times[11] + barOffset, '1.06x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[11])


# A100 GPT2-XL 1024
start4 = 29.5
plt.bar(start4, times[12], color=colors[4], width=default_width, edgecolor='black', label = 'Out of Memory')
plt.text(start4 - 0.1, 0.001, 'OOM', color = 'Red', rotation = 'vertical', fontsize = '28')

plt.bar(start4 + 2 - 0.1, times[13], color=colors[1], width=default_width, edgecolor='black')
print('speedup:', times[12] / times[13])

plt.bar(start4 + 4 - 0.1, times[14], color=colors[2], width=default_width, edgecolor='black')
print('speedup:', times[12] / times[14])

plt.bar(start4 + 6 - 0.1, times[15], color=colors[3], width=default_width, edgecolor='black')
print('speedup:', times[12] / times[15])

plt.ylabel('Time (ms)', fontsize=32)
plt.xlabel('Sequence Length', fontsize=32)
plt.xticks([2.9, 12.3, 23.1, 32.5],['512','1024','512','1024'], fontsize=32)
plt.yticks(fontsize=32)
plt.text(4.7, .023, 'GPT-2', fontsize = 32)
plt.text(24, .023, 'GPT-2 XL', fontsize = 32)
plt.axvline(x=17.7, color = 'black', linestyle = '-.')
plt.margins(0.05, 1.1) 

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(loc='upper left', fontsize=25, frameon=True)

#plt.legend(['_','Full Parameter', 'LoRA', 'Adapter', 'BitFit', 'Out of Memory'], loc='upper left', fontsize=10, frameon=True)
plt.savefig('./graph_scheduler.pdf')
