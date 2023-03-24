import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample arbitrary 2d gaussian distribution with numpy
samples = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[3, 2], [2, 3]],
    size=1000
)
# Plot a kde plot with seaborn
fig = plt.figure()
sns.kdeplot(x=samples[:, 0], y=samples[:, 1])
# Compute the covariance
cov = np.cov(samples, rowvar=False)
# Compute the axis of maximum variance (the eigenvector corresponding to the largest eigenvalue)
eigenvalues, eigenvectors = np.linalg.eig(cov)
eigenvectors = eigenvectors.T
axis_of_max_variance = eigenvectors[np.argmax(eigenvalues)]
# Draw a vector along this axis from the mean
mean = np.mean(samples, axis=0)
plt.arrow(mean[0], mean[1], axis_of_max_variance[0]*5, axis_of_max_variance[1]*5, color="red")
# Sample a continuous MCMV query
def sample_continuous_mcmv_query(samples):
    # Compute the covariance
    cov = np.cov(samples, rowvar=False)
    # Compute the axis of maximum variance (the eigenvector corresponding to the largest eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T
    axis_of_max_variance = eigenvectors[np.argmax(eigenvalues)]
    # Compute the mean of the samples
    mean = np.mean(samples, axis=0)
    # Sample two random points along the axis of max variance that are equidistance from the mean
    distance_scalar = 1.0
    point_1 = mean + distance_scalar * axis_of_max_variance
    point_2 = mean - distance_scalar * axis_of_max_variance

    return (point_1, point_2)
    
query_a, query_b = sample_continuous_mcmv_query(samples)
plt.scatter(query_a[0], query_a[1], color="green")
plt.scatter(query_b[0], query_b[1], color="green")

plt.savefig("continuous_mcmv.png")