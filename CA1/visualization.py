import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.manifold import TSNE
from sklearn.metrics import  accuracy_score

# visualize cross-validation params
# plot the mean cross-validation scores
def plot_cvscores_RBF(scores, params):
    scores_image = mglearn.tools.heatmap(scores, xlabel='num_neurons', xticklabels=params['num_neurons'],
                      ylabel='gamma', yticklabels=params['gamma'], cmap="viridis")

    plt.colorbar(scores_image)
    plt.show()

def plot_cvscores_SVM(scores, params):
    scores_image = mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=params['gamma'],
                      ylabel='C', yticklabels=params['C'], cmap="viridis")

    plt.colorbar(scores_image)
    plt.show()

# visualize training set results
def plot_tsne(data, label):
    X = data
    y = label
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel('1st_component')
    plt.ylabel('2nd_component')
    plt.title('Predicted testing data')
    plt.show()

def plot_decision(data, label, model):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data)
    x = X_tsne[:, 0]
    y = X_tsne[:, 1]
    model.fit(X_tsne, label)
    def make_meshgrid(x, y , h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of SVM classifier ')
    # Set-up grid for plotting.
    xx, yy = make_meshgrid(x, y)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x, y, c=label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('1st features')
    ax.set_xlabel('2nd features')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_leaning(data, label, model):
    X_train = data
    y_train = label.ravel()
    scores = []
    for m in range(2,330):#循环2-79
        model.fit(X_train[:m, :], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        scores.append(accuracy_score(y_train_predict,y_train[:m]))
    plt.plot(range(2,330),scores,c='green', alpha=0.6)
    plt.show()