import matplotlib.pyplot as plt


data = [
    # A100_80GB GPT2 512
    {"task": "a100-1.3b-512-4-torch_full",      "time": 6.467952623367309},
    {"task": "a100-1.3b-512-4-torch_lora",      "time": 0.7862886428833008},
    {"task": "a100-1.3b-512-4-torch_adapter",   "time": 0.13869055807590486},
    {"task": "a100-1.3b-512-4-torch_bitfit",     "time": 0.06162431880831718},
    # A100_80GB GPT2 1024
    {"task": "a100-1.3b-1024-4-torch_full",      "time": 6.448680953979492}, 
    {"task": "a100-1.3b-1024-4-torch_lora",      "time": 0.8067072010040284}, 
    {"task": "a100-1.3b-1024-4-torch_adapter",   "time": 0.1448550409078598}, 
    {"task": "a100-1.3b-1024-4-torch_bitfit",     "time": 0.060682240799069406}, 
    # A100_80GB GPT2-XL 512
    {"task": "a100-2.7b-512-4-torch_full",      "time": 79.87472381591797},
    {"task": "a100-2.7b-512-4-torch_lora",      "time": 2.6044620943069456},
    {"task": "a100-2.7b-512-4-torch_adapter",   "time": 0.7896678388118744}, 
    {"task": "a100-2.7b-512-4-torch_bitfit",     "time": 0.22247423708438874}, 
    # A100_80GB GPT2-XL 1024
    {"task": "a100-2.7b-1024-4-torch_full",      "time": 0},
    {"task": "a100-2.7b-1024-4-torch_lora",      "time": 2.3872102427482607},
    {"task": "a100-2.7b-1024-4-torch_adapter",   "time": 0.8004198443889617},
    {"task": "a100-2.7b-1024-4-torch_bitfit",     "time": 0.2198732775449753},
]

tasks = [d['task'] for d in data]
times = [d['time'] for d in data]

default_width = 1.4
gap_width = 0.2
fontString = '21'
barOffset = 2

plt.figure(figsize=(15, 10))
plt.title("Batch Runtime for One Optimizer Pass", fontsize = 36)
colors=['#0C408C', '#8186D8', '#BF84BA', '#FFDFD3', '#E4080A']

# A100 GPT2 512
plt.bar(0, times[0], color=colors[0], width=default_width, edgecolor='black', label = 'Full Parameter')

plt.bar(2 - 0.1 , times[1], color=colors[1], width=default_width, edgecolor='black', label = 'LoRA')
plt.text(2 - 0.5, times[1] + barOffset, '8.23x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[1])

plt.bar(4 - 0.1, times[2], color=colors[2], width=default_width, edgecolor='black', label = 'Adapter')
plt.text(4 - 0.5, times[2] + barOffset, '46.64x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[2])

plt.bar(6 - 0.1, times[3], color=colors[3], width=default_width, edgecolor='black', label = 'BitFit')
plt.text(6 - 0.5, times[3] + barOffset, '104.96x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[0] / times[3])


# A100 GPT2 1024
start2 = 9.3
plt.bar(start2, times[4], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start2 + 2 - 0.1, times[5], color=colors[1], width=default_width, edgecolor='black')
plt.text(start2 +  2 - 0.5, times[5] + barOffset, '7.99x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[5])

plt.bar(start2 + 4 - 0.1, times[6], color=colors[2], width=default_width, edgecolor='black')
plt.text(start2 + 4 - 0.5, times[6] + barOffset, '44.52x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[6])

plt.bar(start2 + 6 - 0.1, times[7], color=colors[3], width=default_width, edgecolor='black')
plt.text(start2 + 6 - 0.5, times[7] + barOffset, '106.27x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[4] / times[7])


# A100 GPT2-XL 512
start3 = 20.2
plt.bar(start3, times[8], color=colors[0], width=default_width, edgecolor='black')

plt.bar(start3 + 2 - 0.1, times[9], color=colors[1], width=default_width, edgecolor='black')
plt.text(start3 + 2 - 0.5, times[9] + barOffset, '30.67x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[9])

plt.bar(start3 + 4 - 0.1, times[10], color=colors[2], width=default_width, edgecolor='black')
plt.text(start3 + 4 - 0.5, times[10] + barOffset, '101.15x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[10])

plt.bar(start3 + 6 - 0.1, times[11], color=colors[3], width=default_width, edgecolor='black')
plt.text(start3 + 6 - 0.5, times[11] + barOffset, '359.03x', color = 'Black', fontsize = fontString, weight = 'bold', rotation = 'vertical')
print('speedup:', times[8] / times[11])


# A100 GPT2-XL 1024
start4 = 29.5
plt.bar(start4, times[12], color=colors[4], width=default_width, edgecolor='black', label = 'Out of Memory')
plt.text(start4 - 0.1, 2, 'OOM', color = 'Red', rotation = 'vertical', fontsize = '28')

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
plt.text(6, 45, 'GPT-2', fontsize = 32)
plt.text(22, 45, 'GPT-2 XL', fontsize = 32)
plt.axvline(x=17.7, color = 'black', linestyle = '-.')

plt.gca().axes.get_xaxis().set_visible(True)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.legend(loc='upper left', fontsize=25, frameon=True)

#plt.legend(['_','Full Parameter', 'LoRA', 'Adapter', 'BitFit', 'Out of Memory'], loc='upper left', fontsize=10, frameon=True)
plt.savefig('./graph_optimizer.pdf')
