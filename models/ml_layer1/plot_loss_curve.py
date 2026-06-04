"""
STEP3: ML Models - Layer 1 (Body Similarity)
UTILITY: Plot Training Loss Curve (THIS FILE)

PURPOSE: Visualization script for model training progress.
Generates loss curve plots from training logs.
Helps monitor convergence and identify potential training issues.
"""

import matplotlib.pyplot as plt

# Documented training loss values
epochs = [1, 5, 10, 20, 30, 40, 50]
losses = [0.0184, 0.0035, 0.0029, 0.0020, 0.0016, 0.0016, 0.0012]

plt.figure(figsize=(8, 5))

plt.plot(epochs, losses, marker="o", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve for Pose Encoder")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pose_encoder_loss_curve.png", dpi=300)
plt.show()