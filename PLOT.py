

import matplotlib.pyplot as plt

# File Paths and Reading the Log File
log_file_path = "/home/mehrdad/Codes/secondTry/results/Log_2024-07-08_18-32-54/training.txt"  
with open(log_file_path, "r") as log_file:
    lines = log_file.readlines()[1:]  # Skip the header line

# Data Extraction and Processing
epochs = []
steps = []
losses = []
accuracies = []
# Validation losses and accuracies
val_losses = []
val_accuracies = []

current_epoch = 0  # Track the current epoch
for line in lines:
    values = [part for part in line.split() if part]  # Filter out empty strings

    # Ensure line has enough values and the loss is not 'nan'
    if len(values) >= 6 and values[2][2:] != 'nan':
        try:
            # Extract epoch, step, loss, and accuracy
            epoch = int(values[0][1:])
            step = int(values[1][1:])
            loss = float(values[2][2:]) 
            acc = float(values[3][:-1]) / 100
        except ValueError:
            print(f"Warning: Skipping line with invalid format: {line}")
            continue
    else:
        # Check for validation line and extract loss and accuracy
        if len(values) >= 3 and values[0] == 'Validation':
            try:
                val_losses.append(float(values[-5]))
                val_accuracies.append(float(values[-2][:-1]) / 100)
            except ValueError:
                print(f"Warning: Skipping invalid validation line: {line}")
            continue
        else:
            print(f"Warning: Skipping line with invalid format: {line}")
            continue


    # Append to the correct lists based on epoch
    if epoch == current_epoch:
        steps.append(step)
        losses.append(loss)
        accuracies.append(acc)
    else:
        # Plot the previous epoch's data
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label='Train Loss')
        plt.plot(steps, accuracies, label='Train Accuracy')
        if current_epoch < len(val_losses):
            plt.plot(range(current_epoch + 1), val_losses[:current_epoch + 1], label='Validation Loss')
            plt.plot(range(current_epoch + 1), val_accuracies[:current_epoch + 1], label='Validation Accuracy')
        plt.title(f'Training and Validation Losses (Epoch {current_epoch})')
        plt.xlabel('Steps (for training) / Epochs (for validation)')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        plt.show()

        # Reset for the new epoch
        current_epoch = epoch
        steps = [step]
        losses = [loss]
        accuracies = [acc]
        

# Plot the last epoch's data
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label='Train Loss')
plt.plot(steps, accuracies, label='Train Accuracy')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
plt.title(f'Training and Validation Losses (Epoch {current_epoch})')
plt.xlabel('Steps (for training) / Epochs (for validation)')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()