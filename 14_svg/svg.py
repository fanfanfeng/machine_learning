# create by fanfan on 2017/9/9 0009
import numpy as np


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSIM(inA,inB):
    return 1.0/(1.0 + np.linalg.norm(inA - inB))

def prearsSim(inA,inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5 + 0.5*(num/denom)


def standEst(dataMat,user,simMeas,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating =dataMat[user,j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A > 0,dataMat[:,j].A>0))[0]
        if len(overLap) ==0:
            similary = 0
        else:
            similary = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print('the %d and %d similaryitye is :%f' %(item,j,similary))
        simTotal += similary
        ratSimTotal += similary * userRating
    if simTotal == 0:
        return  0
    else:
        return ratSimTotal/simTotal

def recomment(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:
        return "you raed every thing"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return  sorted(itemScores,key=lambda jj:jj[1],reverse =True)[:N]

def svdEst(dataMat,user,simMeas,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xfromedItems = dataMat.T * U[:,:4]* Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j ==item:
            continue
        similarity = simMeas(xfromedItems[item,:].T,xfromedItems[j,:].T)
        print("the %d and %d similarity is :%f" % (item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print("1,",end="")
            else:
                print("0,",end="")
        print("\n")

def imgCompress(numSv =3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat  = np.mat(myl)
    print("**********iroginal matrix *********")
    printMat(myMat,thresh)
    U,Sigma,VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSv,numSv)))
    for k in range(numSv):
        SigRecon[k,k] = Sigma[k]
        reconMat = U[:,:numSv]* SigRecon * VT[:numSv,:]
        print("********** reconstructed matrix using %d singular values ******" % numSv)
        printMat(reconMat,thresh)

if __name__ == '__main__':
    #myMat = np.mat(loadExData2())
    #print(recomment(myMat,2,estMethod=svdEst))
    imgCompress()
