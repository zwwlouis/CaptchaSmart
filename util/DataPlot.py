import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy import *
import numpy as np


def matScatter(data, height, width, name, size=100):
    data = array(data)
    if data.size != (height * width):
        print("输入尺寸有误!")
        return
    data = data.reshape(height, width)
    x = arange(width)
    y = arange(height)
    xx, yy = meshgrid(x, y)

    data_low = (data < 0.1) * size
    plt.scatter(xx, yy, s=data_low, marker='s')
    data_medium = (data > 0.1) * (data < 0.9) * size
    plt.scatter(xx, yy, s=data_medium, marker='s')
    data_high = (data > 0.9) * size
    plt.scatter(xx, yy, s=data_high, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(name)
    plt.legend(["low", "medium", "high"])
    plt.show()


def cmImg(data, name, cmap=None, vmin=None, vmax=None):
    cmImgSub(data, name, 111, cmap, vmin, vmax, show=True)


def cmImgSub(data, name, subarg, cmap, vmin=None, vmax=None, show=False):
    data = array(data)
    if len(data.shape) == 1:
        data = array([data])
    if isinstance(subarg, tuple):
        ax = plt.subplot(subarg[0], subarg[1], subarg[2])
    else:
        ax = plt.subplot(subarg)

    gca = plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    # plt.ylabel('y')
    # plt.xlabel('x')
    plt.title(name)
    ax.set_yticks([])
    if show:
        if data.shape[1] <= 20:
            xLoc = MultipleLocator(1)
            ax.xaxis.set_major_locator(xLoc)
        if vmin is None and vmax is None:
            plt.colorbar(gca, orientation='horizontal')
        plt.show()


def main():
    # 主函数
    print("hello world")
    a = random.rand(20)
    matScatter(a, 4, 5, "test", size=500)


if __name__ == "__main__":
    main()
