import seaborn as sns
import matplotlib.pyplot as plt

def plot_chart_all(dataset):
    figure, axis = plt.subplots(3, 3, figsize=(16, 16))

    sns.histplot(data=dataset, x='I0', kde=True, ax=axis[0, 0])
    sns.histplot(data=dataset, x='PA500', kde=True, ax=axis[0, 1])
    sns.histplot(data=dataset, x='HFS', kde=True, ax=axis[0, 2])

    sns.histplot(data=dataset, x='DA', kde=True, ax=axis[1, 0])
    sns.histplot(data=dataset, x='Area', kde=True, ax=axis[1, 1])
    sns.histplot(data=dataset, x='A/DA', kde=True, ax=axis[1, 2])

    sns.histplot(data=dataset, x='Max IP', kde=True, ax=axis[2, 0])
    sns.histplot(data=dataset, x='DR', kde=True, ax=axis[2, 1])
    sns.histplot(data=dataset, x='P', kde=True, ax=axis[2, 2])

    figure.suptitle('Data Distribution')

def plot_chart_numerical(dataset):
    figure, axis = plt.subplots(3, 3, figsize=(16, 16))

    sns.boxplot(data=dataset, x='I0', hue='Class', ax=axis[0, 0])
    sns.boxplot(data=dataset, x='PA500', hue='Class', ax=axis[0, 1])
    sns.boxplot(data=dataset, x='HFS', hue='Class', ax=axis[0, 2])

    sns.boxplot(data=dataset, x='DA', hue='Class', ax=axis[1, 0])
    sns.boxplot(data=dataset, x='Area', hue='Class', ax=axis[1, 1])
    sns.boxplot(data=dataset, x='A/DA', hue='Class', ax=axis[1, 2])

    sns.boxplot(data=dataset, x='Max IP', hue='Class', ax=axis[2, 0])
    sns.boxplot(data=dataset, x='DR', hue='Class', ax=axis[2, 1])
    sns.boxplot(data=dataset, x='P', hue='Class', ax=axis[2, 2])

    figure.suptitle('Data Distribution (Box Plot)')

def plot_chart_by_class(dataset):
    figure, axis = plt.subplots(3, 3, figsize=(16, 16))

    sns.boxplot(data=dataset, x='I0', y='Class', ax=axis[0, 0])
    sns.boxplot(data=dataset, x='PA500', y='Class', ax=axis[0, 1])
    sns.boxplot(data=dataset, x='HFS', y='Class', ax=axis[0, 2])

    sns.boxplot(data=dataset, x='DA', y='Class', ax=axis[1, 0])
    sns.boxplot(data=dataset, x='Area', y='Class', ax=axis[1, 1])
    sns.boxplot(data=dataset, x='A/DA', y='Class', ax=axis[1, 2])

    sns.boxplot(data=dataset, x='Max IP', y='Class', ax=axis[2, 0])
    sns.boxplot(data=dataset, x='DR', y='Class', ax=axis[2, 1])
    sns.boxplot(data=dataset, x='P', y='Class', ax=axis[2, 2])

    figure.suptitle('Data Distribution by Class')

def plot_chart_by_IO(dataset):
    # sns.set_style('whitegrid', {'axes.grid' : False})
    sns.catplot(data=dataset, x='Class', y='I0', kind='swarm', height= 5, aspect= 2, s= 20)

    plt.title('Data Distribution by IO')

def plot_chart_by_PA500(dataset):
    sns.catplot(data=dataset, x='Class', y='PA500', kind='swarm', height= 5, aspect= 2, s= 20)

    plt.title('Data Distribution by PA500')
