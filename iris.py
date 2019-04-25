import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import minmax_scale

iris = sns.load_dataset('iris')
# iris = iris.loc[iris['species'] != 'virginica']
iris = iris.drop(['sepal_width', 'petal_width'], axis=1)
iris.loc[iris['species'] == 'virginica', 'petal_length'] += 2

# X = iris[['sepal_length', 'petal_length']].to_numpy()
# X = minmax_scale(X)

# y = iris['species'].to_numpy()
# c = {'setosa': 0, 'versicolor': 1}
# y = [c[i] for i in y]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

sns.pairplot(iris, hue='species')
plt.show()