import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Generate data for two Gaussian distributions
# Using fixed random state for reproducibility
np.random.seed(0)
data1 = np.random.normal(loc=0, scale=1, size=1000)
data2 = np.random.normal(loc=3, scale=1.5, size=1000)

# Create a pandas DataFrame
df = pd.DataFrame({
    'Value': np.concatenate([data1, data2]),
    'Distribution': ['Distribution 1 (μ=0, σ=1)'] * 1000 + ['Distribution 2 (μ=3, σ=1.5)'] * 1000
})

# Create a KDE plot
plt.figure(figsize=(15, 8))
sns.kdeplot(data=df, x='Value', hue='Distribution', fill=True, common_norm=False, alpha=0.5)

# --- Interval calculations ---

# 1. Overlapping interval
overlap_lower = np.percentile(data1, 95)
overlap_upper = np.percentile(data2, 5)

# 2. Outer-union intervals (boundaries for the "outside" regions)
outside_lower = np.percentile(data1, 5)
outside_upper = np.percentile(data2, 95)


# --- Coloring the intervals ---

# Get the current axes
ax = plt.gca()
# Get the x-axis limits of the plot for the fill
xmin, xmax = ax.get_xlim()

# Color the intervals using axvspan for simplicity and clarity
# Region 1: Overlapping interval
ax.axvspan(overlap_lower, overlap_upper, color='green', alpha=0.2, label='Low Anomaly')

# Region 2: The value lies outside the overlapping region but within the union of both distributions
ax.axvspan(outside_lower, overlap_lower, color='yellow', alpha=0.2, label='Medium Anomaly')
ax.axvspan(overlap_upper, outside_upper, color='yellow', alpha=0.2)


# Region 3: The value falls outside the lower group’s 5th percentile or the higher group’s 95th percentile.
ax.axvspan(xmin, outside_lower, color='red', alpha=0.2, label='High Anomaly')
ax.axvspan(outside_upper, xmax, color='red', alpha=0.2)


# Add vertical lines to show the exact percentile points
plt.axvline(outside_lower, color='black', linestyle='--', label='5th Percentile (lower group)')
plt.axvline(overlap_lower, color='blue', linestyle='--', label='95th Percentile (lower group)')
plt.axvline(overlap_upper, color='purple', linestyle='--', label='5th Percentile (upper group)')
plt.axvline(outside_upper, color='brown', linestyle='--', label='95th Percentile (upper group)')


# Add labels, a title, and a legend
plt.xlabel('Value', fontsize=14, fontweight='bold')
plt.ylabel('Density', fontsize=14, fontweight='bold')
plt.title('Two distributions with colored intervals', fontsize=16, fontweight='bold')

# Create a new legend that combines the KDE and the fill patches
handles, labels = ax.get_legend_handles_labels()
# Get the KDE handles (first two items)
kde_handles = handles[:2]
kde_labels = labels[:2]
# Create patch handles for the colored regions
patch_handles = [
    Patch(facecolor='green', alpha=0.2, label='Low anomaly'),
    Patch(facecolor='yellow', alpha=0.2, label='Medium anomaly'),
    Patch(facecolor='red', alpha=0.2, label='High anomaly')
]
# Add percentile line handles
line_handles = handles[3:]
line_labels = labels[3:]


ax.legend(handles=patch_handles + line_handles, labels=[h.get_label() for h in patch_handles] + line_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig('gaussian_intervals_colored.pdf', format='pdf')
plt.show()