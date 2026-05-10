import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Define column names based on standard CMAPSS dataset structure
columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
for i in range(1, 22):
    columns.append(f's_{i}')

# Load the data
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns)

# Drop columns with constant values as they have no skewness
df = df.loc[:, df.nunique() > 1]

# Calculate skewness for all remaining columns
skewness = df.skew().sort_values(ascending=False)

print("Skewness of each feature:")
print(skewness.to_string())

# Plot skewness bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=skewness.index, y=skewness.values, palette='coolwarm')
plt.xticks(rotation=90)
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(1, color='red', linestyle='--', label='Highly Skewed (>1 or <-1)')
plt.axhline(-1, color='red', linestyle='--')
plt.title('Skewness of Features in CMAPSS train_FD001 Data')
plt.ylabel('Skewness')
plt.xlabel('Features')
plt.legend()
plt.tight_layout()
plt.savefig('skewness_bar_plot.png')
print("\nSaved skewness_bar_plot.png")

# Identify highly skewed and relatively normal features
highly_skewed = skewness[abs(skewness) > 1].index.tolist()
normal_features = skewness[abs(skewness) < 0.5].index.tolist()

print(f"\nHighly skewed features (> |1|): {highly_skewed}")
print(f"Relatively normal features (< |0.5|): {normal_features}")

# Plot distributions for a few selected features to visualize the skewness
features_to_plot = []
if highly_skewed:
    features_to_plot.append(skewness.index[0])  # Most positively skewed
    if skewness.index[-1] in highly_skewed:
        features_to_plot.append(skewness.index[-1]) # Most negatively skewed
if normal_features:
    features_to_plot.append(normal_features[len(normal_features)//2]) # Pick a normal one

# Ensure unique and limit to a few
features_to_plot = list(dict.fromkeys(features_to_plot))[:3]

if features_to_plot:
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features_to_plot):
        plt.subplot(1, len(features_to_plot), i + 1)
        sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}\nSkew: {skewness[feature]:.2f}')
    
    plt.tight_layout()
    plt.savefig('skewness_distributions.png')
    print("Saved skewness_distributions.png")
