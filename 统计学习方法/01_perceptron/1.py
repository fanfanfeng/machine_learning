# create by fanfan on 2018/3/13 0013
import copy
from matplotlib import pyplot as plt
from matplotlib import animation
training_set = [[(3,3),1],[(4,3),1],[(1,1),-1]]
w = [0,0]
b = 0
history = []
def update(item):
    """
        update parameters using stochastic gradient descent
        :param item: an item which is classified into wrong class
        :return: nothing
        """
    global  w,b,history
    w[0] += 1* item[1] * item[0][0]
    w[1] += 1* item[1]* item[0][1]
    b += 1* item[1]
    print(w,b)
    history.append([copy.copy(w),b])

def cal(item):
    """
       calculate the functional distance between 'item' an the dicision surface. output yi(w*xi+b).
       :param item:
       :return:
       """
    res = 0
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    res *= item[1]
    return res

def check():
    """
        check if the hyperplane can classify the examples correctly
        :return: true if it can
        """
    flag = False
    for item in training_set:
        if cal(item) <=0:
            flag = True
            update(item)

    if not flag:
        print("RESULT:w:"+str(w)+ "b:"+str(b))
    return flag

if __name__ == '__main__':
    for i in range(1000):
        if not check():
            break

    fig = plt.figure()
    ax = plt.axes(xlim=(0,2),ylim=(-2,2))
    line, = ax.plot([],[],'g',lw=2)
    label = ax.text([],[],'')

    def init():
        line.set_data([],[])
        x,y,x_,y_ = [],[],[],[]
        for p in training_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])

        plt.plot(x,y,'bo',x_,y_,'rx')
        plt.axis([-6,6,-6,6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Perceptron algorithm')
        return line,label

    def animate(i):
        global history,ax,line,label

        w = history[i][0]
        b = history[i][1]

        if w[1] == 0:
            return line,label

        x1 = -7
        y1 = -(b + w[0]*x1)/w[1]
        x2 = 7
        y2 = -(b +w[0]*x2)/w[1]
        line.set_data([x1,x2],[y1,y2])
        x1 = 0
        y1 = -(b+w[0] * x1)/w[1]
        label.set_text(history[i])
        label.set_position([x1,y1])
        return line,label
    print(history)
    animate(1)
    anim = animation.FuncAnimation(fig,animate,init_func=init,
                                   frames=len(history),interval=1000,blit=True)
    plt.show()
    #anim.save('perceptron.gif',fps=2,writer='imagemagick')
