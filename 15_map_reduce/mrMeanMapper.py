# create by fanfan on 2017/9/9 0009
import numpy as np
import sys
def read_input(file):
    for line in file:
        yield line.rstrip()

input = read_input(sys.stdin)
input = [float(line) for line in input if line!=""]
numInputs = len(input)
input = np.mat(input)
sqInput = np.power(input,2)
#output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput)))#calc mean of columns
print( "report: still alive",file=sys.stderr)