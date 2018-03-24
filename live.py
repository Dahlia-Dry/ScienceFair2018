import numpy as np
from time import sleep
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AnalogPlot:
    def __init__(self, maxLen):
        self.ax = deque([0.0]*maxLen)
        self.ay = deque([0.0]*maxLen)
        self.az = deque([0.0]*maxLen)
        self.maxLen = maxLen
        self.load = np.loadtxt('tests/data/rawData/0raw8.txt')

    def addToBuffer(self, buffer, val):
        if len(buffer) < self.maxLen:
            buffer.append(val)
        else:
            buffer.pop()
            buffer.appendleft(val)

    def add(self, data):
        assert(len(data) == 1)
        self.addToBuffer(self.ax, data[0])
        np.delete(data, 0)
		
    def update(self, frameNum, a0):
        try:
            data = self.load
            if(len(data) == 1):
                self.add(data)
                a0.set_data(range(self.maxLen), self.ax)
        except KeyboardInterrupt:
            print('exiting')
        return a0,
   


def main():
    load = np.loadtxt('tests/data/rawData/0raw8.txt')
    analogPlot = AnalogPlot(350)

    print('plotting data...')

    # set up animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 360), ylim=(np.amin(load), np.amax(load)))
    a0, = ax.plot([], [])

    anim = animation.FuncAnimation(fig, analogPlot.update,
                                   fargs=(a0),
                                   interval=50)

    # show plot
    plt.show()


    print('exiting.')

main()