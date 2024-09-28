import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load the JSON data
with open('output/all_gmm_params.json', 'r') as f:
    data = json.load(f)

# Extract means and covariances
means_x = []
means_y = []
cov_xx = []
cov_yy = []
cov_xy = []

for image, components in data.items():
    for component in components.values():
        means = component['means']
        covariance = component['covariances']
        means_x.append(means[0])
        means_y.append(means[1])
        cov_xx.append(covariance[0][0])
        cov_yy.append(covariance[1][1])
        cov_xy.append(covariance[0][1])

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(20, 15))
fig.suptitle('Distributions of Means and Covariances', fontsize=16)

# Plot distribution of means
sns.kdeplot(means_x, ax=axs[0, 0], shade=True)
axs[0, 0].set_title('Distribution of Means (X)')
axs[0, 0].set_xlabel('X')

sns.kdeplot(means_y, ax=axs[0, 1], shade=True)
axs[0, 1].set_title('Distribution of Means (Y)')
axs[0, 1].set_xlabel('Y')

# Scatter plot of means
axs[0, 2].scatter(means_x, means_y, alpha=0.5)
axs[0, 2].set_title('Scatter Plot of Means')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')

# Plot distribution of covariances
sns.kdeplot(cov_xx, ax=axs[1, 0], shade=True)
axs[1, 0].set_title('Distribution of Covariances (XX)')
axs[1, 0].set_xlabel('Covariance XX')

sns.kdeplot(cov_yy, ax=axs[1, 1], shade=True)
axs[1, 1].set_title('Distribution of Covariances (YY)')
axs[1, 1].set_xlabel('Covariance YY')

sns.kdeplot(cov_xy, ax=axs[1, 2], shade=True)
axs[1, 2].set_title('Distribution of Covariances (XY)')
axs[1, 2].set_xlabel('Covariance XY')

plt.tight_layout()
plt.savefig('gmm_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Create heatmap of mean correlations
mean_data = defaultdict(list)
for image, components in data.items():
    image_means = [component['means'] for component in components.values()]
    mean_data[image] = np.mean(image_means, axis=0)

mean_matrix = np.array(list(mean_data.values()))
correlation_matrix = np.corrcoef(mean_matrix.T)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Mean GMM Components Across Images')
plt.tight_layout()
plt.savefig('gmm_mean_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved as 'gmm_distributions.png' and 'gmm_mean_correlation_heatmap.png'")
