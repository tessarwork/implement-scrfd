import matplotlib.pyplot as plt
import re
import os
log_file = "logs/train_2025-12-04_07-56-39.log"

pattern_trainloss = r"Epoch (\d+) Done \| Train Loss: ([\d\.]+) \| Last LR: ([\d\.]+)"
pattern_valloss = r"Epoch (\d+) Validation \| Loss: ([\d\.]+) \| Cls: ([\d\.]+) \| Reg: ([\d\.]+)"

epochs = []
train_losses = []
val_epochs, val_losses = [], []
lrs = []

if not os.path.exists(log_file): 
    print(f"ERROR: File '{log_file}' tidak ditemukan!")
    # Gunakan dummy data jika file tidak ada untuk demonstrasi
    lines = []
else: 
    print(f"Membaca file: {log_file}...")
    with open(log_file, 'r') as f:
        lines = f.readlines()
for line in lines: 
    train_match = re.search(pattern_trainloss, line)
    if train_match:
        epochs.append(int(train_match.group(1)))
        train_losses.append(float(train_match.group(2)))
        lrs.append(float(train_match.group(3)))
        continue
    val_match = re.search(pattern_valloss, line)
    if val_match: 
        val_epochs.append(int(val_match.group(1)))
        val_losses.append(float(val_match.group(2)))

print(f"Total Data Training ditemukan: {len(epochs)}")
# print(f"Total Data Training ditemukan: {len(train_epochs)}")
print(f"Total Data Validation ditemukan: {len(val_epochs)}")

if len(epochs) == 0: 
    print("WARNING: Tidak ada data yang terdeteksi. Cek format regex atau isi file log.")
else: 
    # --- 4. VISUALISASI ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylim(0, 2.0)

    # Plot Loss (Train vs Val)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    # Train Loss (Merah Putus-putus)
    ax1.plot(epochs, train_losses, color='tab:red', linestyle='--', alpha=0.6, label='Train Loss')
    # Val Loss (Merah Tebal - Lebih penting)
    if val_epochs:
        ax1.plot(val_epochs, val_losses, color='darkred', marker='o', markersize=4, label='Validation Loss')
    
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Learning Rate (Sumbu Kanan - Biru)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue')
    ax2.plot(epochs, lrs, color='tab:blue', linestyle='-', alpha=0.3, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Training Progress: Train Loss vs Validation Loss')
    plt.tight_layout()
    plt.show()

#     match_train = re.search(pattern_trainloss, line)
#     if match_train: 
#         epochs.append(int(match_train.group(1)))
#         train_losses.append(float(match_train.group(2)))
#         lrs.append(float(match_train.group(3)))

# print(epochs)
# 3. Visualization
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot Train Loss (Left Axis - Red)
# color = 'tab:red'
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Train Loss', color=color)
# ax1.plot(epochs, train_losses, color=color, marker='o')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid(True, alpha=0.3)

# # Plot Learning Rate (Right Axis - Blue)
# ax2 = ax1.twinx()  
# color = 'tab:blue'
# ax2.set_ylabel('Learning Rate', color=color)  
# ax2.plot(epochs, lrs, color=color, linestyle='--', marker='x')
# ax2.tick_params(axis='y', labelcolor=color)

# plt.title('Training Loss & Learning Rate')
# plt.show()