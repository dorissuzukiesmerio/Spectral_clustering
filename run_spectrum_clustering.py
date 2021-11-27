import pandas
import matplotlib.pyplot as pyplot

from sklearn.cluster import SpectralClustering

dataset = pandas.read_csv("dataset_moon.csv")

print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_moon.png")
pyplot.close()

machine = SpectralClustering(n_clusters=2, n_components=4, gamma = 10, affinity="laplacian")
#Gamma = how heavy is the weight on what we decide what dot is to the other.
# N_components = (use initiallly the same as number of clusters. But increase it if )

# Try n_components = 2 and gamma =8 first, then increase
result = machine.fit_predict(dataset)


pyplot.scatter(dataset['x1'], dataset['x2'], c = result)
pyplot.savefig("scatterplot_moon_color.png")
pyplot.close()