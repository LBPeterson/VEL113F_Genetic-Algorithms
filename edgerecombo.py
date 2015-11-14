# pete9348@atlas.cselabs.umn.edu is the server to be run on

import numpy as np

prcro = 0.8
prmut = 0.05
ngen = 200
npop = 200
od = 30
coords = np.array([
[54, 67],
[54, 62],
[37, 84],
[41, 94],
[ 2, 99],
[ 7, 64],
[25, 62],
[22, 60],
[18, 54],
[ 4, 50],
[13, 40],
[18, 40],
[24, 42],
[25, 38],
[44, 35],
[41, 26],
[45, 21],
[58, 35],
[62, 32],
[82,  7],
[91, 38],
[83, 46],
[71, 44],
[64, 60],
[68, 58],
[83, 69],
[87, 76],
[74, 78],
[71, 71],
[58, 69]])


#### FUNCTIONS ####
def createIndividuals(rows, cols):
    x = np.zeros([rows,cols])
    for i in range(0,np.shape(x)[0]):
        x[i,:] = np.random.permutation(cols)
    return x ;


def calculatePI(matrix, coords):
    PI = np.zeros(np.shape(matrix)[0])
    for j in range(0,np.size(PI)):
        for k in range(0,od-1):
            PI[j] += np.sqrt(np.square(coords[x[j][k]][0]-coords[x[j][k+1]][0])+np.square(coords[x[j][k]][1]-coords[x[j][k+1]][1]))
        PI[j] += np.sqrt(np.square(coords[x[j][od-1]][0]-coords[x[j][0]][0])+np.square(coords[x[j][od-1]][1]-coords[x[j][0]][1]))
    return PI ;

def makeEdgeTable(p1,p2):
    edgeTable = np.empty([np.shape(coords)[0],4]) * np.nan
    for j in range(1,od-1):
        point = p1[j]
        index = np.where(np.isnan(edgeTable[point]))[0][0]
        edgeTable[point][index] = p1[j-1]
        edgeTable[point][index+1] = p1[j+1]
        point = p2[j]
        index = np.where(np.isnan(edgeTable[point]))[0][0]
        edgeTable[point][index] = p2[j-1]
        edgeTable[point][index+1] = p2[j+1]
    # when j = 0
    point = p1[0]
    index = np.where(np.isnan(edgeTable[point]))[0][0]
    edgeTable[point][index] = p1[od-1]
    edgeTable[point][index+1] = p1[1]
    point = p2[0]
    index = np.where(np.isnan(edgeTable[point]))[0][0]
    edgeTable[point][index] = p2[od-1]
    edgeTable[point][index+1] = p2[1]
    # when j = od -1
    point = p1[od-1]
    index = np.where(np.isnan(edgeTable[point]))[0][0]
    edgeTable[point][index] = p1[od-2]
    edgeTable[point][index+1] = p1[0]
    point = p2[od-1]
    index = np.where(np.isnan(edgeTable[point]))[0][0]
    edgeTable[point][index] = p2[od-2]
    edgeTable[point][index+1] = p2[0]
    return edgeTable;

def getNextIndex(c,e):
    nexts = np.unique(e[c])
    nextind = np.where(~np.isnan(np.unique(e[c])))[0]
    potential = np.zeros([nextind.size,4])
    score = np.zeros(nextind.size)
    for i in range(0,nextind.size):
        potential[i] = e[nexts[nextind[i]]]
        score[i] = np.where(~np.isnan(np.unique(potential[i])))[0].size;
    if isFullNaN(nexts):
        return -1;
    mins = np.where(score==score.min())[0]
    chosenIndex = np.random.randint(mins.size)
    return nexts[mins[chosenIndex]].astype(int)

def isFullNaN(a):
    if( a.size != 4):
        return False;
    elif( np.isnan(a[0]) & np.isnan(a[1]) & np.isnan(a[2]) & np.isnan(a[3]) ):
        return True;
    else:
        return False;

def recombinateEdges(eT):
    edges = eT
    child = np.empty(od) * np.nan
    # first point selected at random
    current = np.random.randint(od)
    child[0] = current
    for i in range(0,od-1):
        nextIndex = getNextIndex(current,edges)
        if (isFullNaN(edges[nextIndex])) | (nextIndex == -1): #find new next is all nan
            for j in range(0,od):
                if isFullNaN(edges[j]):
                    continue;
                else:
                    break;
            nextIndex = j
        edges[current] = np.empty(4)*np.nan
        toReplace = np.where(edges==current)
        for j in range(0,toReplace[0].size):
            edges[toReplace[0][j]][toReplace[1][j]] = np.nan;
        ##### add to child
        child[np.where(np.isnan(child))[0][0]] = nextIndex
        current = nextIndex
    return child

def reciprocalMutation(array):
    i1 = np.random.randint(od)
    i2 = np.random.randint(od)
    temp = array[i1]
    array[i1] = array[i2]
    array[i2] = temp
    return array;


### CREATE INITIAL GENERATION
x = createIndividuals(npop,od)
endX = np.zeros([ngen,od])
endPI = np.zeros([ngen])


for h in range(0,ngen):
    PI = calculatePI(x,coords)
    ng = np.zeros(x.shape)
    ### EDGE RECOMBINATION
    for i in range(0,npop):
        p1 = np.random.randint(npop)
        p2 = np.random.randint(npop)
        if np.random.rand() < prcro:
            eT = makeEdgeTable(x[p1],x[p2])
            ng[i] = recombinateEdges(eT)
        elif PI[p1]<PI[p2]:
            ng[i] = x[p1]
        else:
            ng[i] = x[p2]
        #This is a quick fix for the edgerecombo missing the last element sometimes
        if np.unique(ng[i]).size != od:
            for j in range(0,od):
                if j not in ng[i]:
                    index = np.amax(np.where(ng[i]==od-1)[0]) # finds the last instance of od-1
                    if np.where(ng[i]==od-1)[0].size >=3:
                        print(np.where(ng[i]==od-1)[0].size)
                    ng[i][index] = j


    ### RECIPROCAL MUTATION
    for i in range(0,npop):
        if np.random.rand() < prmut:
            ng[i] = reciprocalMutation(ng[i])


    ### SWAP BEST OLD VALUE
    PI = calculatePI(x,coords)
    ngPI = calculatePI(ng,coords)
    oldMin = np.argmin(PI)
    newMax = np.argmax(ngPI)
    ng[newMax] = x[oldMin]
    ngPI[newMax] = PI[oldMin]
    x = ng
    PI = ngPI

    endX[h] = x[np.argmin(PI)]
    endPI[h] = np.amin(PI)

    print("Generation: %d" % (h))

print("############")
print("Min PI is: %f" % (np.amin(endPI)))
print(x[np.argmin(endPI)])
