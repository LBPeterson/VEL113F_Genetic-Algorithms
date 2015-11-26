# Luke Peterson 220394-4339
# VEL113F

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

prcro = 0.8
prmut = 0.05
ngen = 20000
npop = 20
od = 132
coords = np.array([         #[lng from, lat from, lng to, lat to]
[-18.2666666667 ,65.9666666667  ,-18.35         ,66.0333333333  ],
[-19.7833333333 ,65.9333333333  ,-19.8333333333 ,66             ],
[-20.5252777778 ,65.7411111111  ,-20.5627777778 ,65.8066666667  ],
[-20.5333333333 ,65.9166666667  ,-20.6027777778 ,65.9861111111  ],
[-20.75         ,65.9833333333  ,-20.8          ,65.9166666667  ],
[-20.9936111111 ,65.9583333333  ,-21.1394444444 ,65.91          ],
[-21.0408333333 ,65.6338888889  ,-21.0919444444 ,65.5611111111  ],
[-16.7083333333 ,66.3758333333  ,-16.7261111111 ,66.4402777778  ],
[-16.8061111111 ,66.2711111111  ,-16.6922222222 ,66.3169444444  ],
[-17.5333333333 ,66.1333333333  ,-17.4333333333 ,66.1972222222  ],
[-17.7          ,66.3333333333  ,-17.7833333333 ,66.4           ],
[-17.6          ,66.45          ,-17.4333333333 ,66.45          ],
[-17.75         ,66.2333333333  ,-17.7166666667 ,66.1666666667  ],
[-17.5          ,66.25          ,-17.5          ,66.3166666667  ],
[-17.3166666667 ,66.25          ,-17.3666666667 ,66.1833333333  ],
[-18.3          ,66.25          ,-18.3833333333 ,66.3           ],
[-18.3666666667 ,66.4333333333  ,-18.3666666667 ,66.5           ],
[-18.45         ,66.3333333333  ,-18.45         ,66.4           ],
[-18.1166666667 ,66.45          ,-18.2333333333 ,66.4           ],
[-18.35         ,66.0833333333  ,-18.4166666667 ,66.15          ],
[-18.8          ,66.3666666667  ,-18.9055555556 ,66.4277777778  ],
[-19.7833333333 ,66.3166666667  ,-19.7736111111 ,66.3875        ],
[-19.6297222222 ,66.3638888889  ,-19.6055555556 ,66.2972222222  ],
[-19.6          ,66.25          ,-19.77         ,66.2305555556  ],
[-19.05         ,66.3333333333  ,-19.0833333333 ,66.4           ],
[-19.7833333333 ,66.0833333333  ,-19.65         ,66.1269444444  ],
[-19.1666666667 ,66.1833333333  ,-19.15         ,66.25          ],
[-19.9          ,66             ,-19.8          ,66.05          ],
[-19.4          ,66.3333333333  ,-19.2333333333 ,66.3333333333  ],
[-19.9          ,66.0833333333  ,-19.85         ,66.15          ],
[-20.6333333333 ,66.1833333333  ,-20.65         ,66.25          ],
[-20.7833333333 ,66.3833333333  ,-20.8          ,66.45          ],
[-20            ,66.2166666667  ,-20            ,66.2833333333  ],
[-20.3          ,66.3333333333  ,-20.3333333333 ,66.4           ],
[-21.0602777778 ,66.4944444444  ,-21.1333333333 ,66.4333333333  ],
[-21.2666666667 ,66.4138888889  ,-21.3405555556 ,66.4761111111  ],
[-21.6055555556 ,66.5           ,-21.7863888889 ,66.5033333333  ],
[-21.4166666667 ,66.4           ,-21.3344444444 ,66.3344444444  ],
[-21.05         ,66.3333333333  ,-21.05         ,66.2666666667  ],
[-21.2833333333 ,66.2833333333  ,-21.15         ,66.2333333333  ],
[-21.25         ,66.15          ,-21.2391666667 ,66.0911111111  ],
[-15.1866666667 ,66.7419444444  ,-15.1866666667 ,66.8025        ],
[-15.2066666667 ,66.9419444444  ,-15.0577777778 ,66.9730555556  ],
[-15.8363888889 ,66.7411111111  ,-15.6708333333 ,66.7333333333  ],
[-15.65         ,66.6838888889  ,-15.5683333333 ,66.6402777778  ],
[-15.9252777778 ,66.8108333333  ,-15.8919444444 ,66.8686111111  ],
[-15.1025       ,66.5847222222  ,-15.2688888889 ,66.575         ],
[-15.4502777778 ,66.5883333333  ,-15.35         ,66.6405555556  ],
[-15.4105555556 ,66.755         ,-15.5605555556 ,66.7219444444  ],
[-15.4038888889 ,66.5069444444  ,-15.5508333333 ,66.535         ],
[-15.4713888889 ,66.9194444444  ,-15.6244444444 ,66.9433333333  ],
[-16.7916666667 ,66.7602777778  ,-16.7736111111 ,66.8363888889  ],
[-16.8416666667 ,66.9186111111  ,-16.6552777778 ,66.9219444444  ],
[-16.5033333333 ,66.7525        ,-16.3380555556 ,66.7547222222  ],
[-16.2533333333 ,66.8380555556  ,-16.1236111111 ,66.8016666667  ],
[-16.8338888889 ,67.0002777778  ,-16.6611111111 ,67.0011111111  ],
[-16.8369444444 ,66.7394444444  ,-16.7969444444 ,66.67          ],
[-16.0447222222 ,66.69          ,-16.0402777778 ,66.7555555556  ],
[-16.9591666667 ,66.6669444444  ,-16.8455555556 ,66.6263888889  ],
[-16.6694444444 ,66.7613888889  ,-16.6088888889 ,66.8338888889  ],
[-16.4327777778 ,66.6616666667  ,-16.2502777778 ,66.6680555556  ],
[-17.2833333333 ,66.6833333333  ,-17.3          ,66.6166666667  ],
[-18            ,66.9166666667  ,-18            ,66.85          ],
[-17.3          ,66.8333333333  ,-17.3          ,66.9           ],
[-17.4          ,66.75          ,-17.2          ,66.75          ],
[-17.05         ,66.75          ,-17.16         ,66.8052777778  ],
[-18.4833333333 ,66.8166666667  ,-18.3166666667 ,66.8333333333  ],
[-18.4666666667 ,66.9833333333  ,-18.4666666667 ,66.9166666667  ],
[-18            ,66.6333333333  ,-18.15         ,66.65          ],
[-18.5166666667 ,66.7           ,-18.4          ,66.75          ],
[-18.4          ,66.5833333333  ,-18.4166666667 ,66.65          ],
[-18.6          ,66.5           ,-18.55         ,66.5666666667  ],
[-18.35         ,66.7           ,-18.25         ,66.75          ],
[-18.5688888889 ,66.7830555556  ,-18.5661111111 ,66.8513888889  ],
[-18.1          ,66.5833333333  ,-18.25         ,66.6166666667  ],
[-18.8          ,66.95          ,-18.85         ,67             ],
[-19.5166666667 ,66.7333333333  ,-19.6333333333 ,66.7833333333  ],
[-19.6666666667 ,66.5833333333  ,-19.5833333333 ,66.65          ],
[-19.8          ,66.65          ,-19.7333333333 ,66.7           ],
[-19.2166666667 ,66.6833333333  ,-19.2833333333 ,66.6166666667  ],
[-19.4          ,66.5           ,-19.3666666667 ,66.5666666667  ],
[-19.35         ,66.75          ,-19.3666666667 ,66.8166666667  ],
[-19.95         ,66.75          ,-19.8          ,66.7833333333  ],
[-20.2166666667 ,66.5666666667  ,-20.0666666667 ,66.6166666667  ],
[-20            ,66.65          ,-20            ,66.7166666667  ],
[-20.8          ,66.8           ,-20.6666666667 ,66.85          ],
[-20.8333333333 ,66.9833333333  ,-20.6833333333 ,66.9833333333  ],
[-20.5          ,66.9333333333  ,-20.3666666667 ,66.9666666667  ],
[-20.9166666667 ,66.9166666667  ,-21            ,66.9666666667  ],
[-20.45         ,66.5833333333  ,-20.2833333333 ,66.6166666667  ],
[-20.3          ,66.6666666667  ,-20.15         ,66.7           ],
[-20.7          ,66.5           ,-20.7166666667 ,66.5666666667  ],
[-21.45         ,66.6666666667  ,-21.2833333333 ,66.6666666667  ],
[-21.1666666667 ,66.8833333333  ,-20.9666666667 ,66.9           ],
[-21.2166666667 ,66.7333333333  ,-21.1333333333 ,66.8           ],
[-21.45         ,66.8           ,-21.2666666667 ,66.8166666667  ],
[-21.9166666667 ,66.6           ,-21.75         ,66.6166666667  ],
[-21.55         ,66.5833333333  ,-21.45         ,66.65          ],
[-21.65         ,66.9166666667  ,-21.4833333333 ,66.9333333333  ],
[-21.7          ,66.75          ,-21.85         ,66.7833333333  ],
[-21.9166666667 ,66.8833333333  ,-21.7666666667 ,66.9166666667  ],
[-21.1          ,66.6666666667  ,-21            ,66.7166666667  ],
[-22.1166666667 ,66.8833333333  ,-22.2166666667 ,66.9333333333  ],
[-22.25         ,66.9833333333  ,-22.4333333333 ,66.9833333333  ],
[-22.4333333333 ,66.95          ,-22.5666666667 ,66.9           ],
[-22.8166666667 ,66.95          ,-22.9666666667 ,66.9           ],
[-22.6833333333 ,66.9333333333  ,-22.5833333333 ,66.9833333333  ],
[-22.4166666667 ,66.8666666667  ,-22.25         ,66.85          ],
[-22.6          ,66.75          ,-22.7166666667 ,66.8           ],
[-22.7          ,66.8333333333  ,-22.5833333333 ,66.8833333333  ],
[-22.3833333333 ,66.5833333333  ,-22.4          ,66.65          ],
[-22.1333333333 ,66.7333333333  ,-22.1          ,66.6666666667  ],
[-22.95         ,66.75          ,-22.8833333333 ,66.8166666667  ],
[-16.9002777778 ,67.1011111111  ,-16.7172222222 ,67.1013888889  ],
[-17.4333333333 ,67             ,-17.3833333333 ,67.0666666667  ],
[-18.5          ,66.9833333333  ,-18.5          ,67.0666666667  ],
[-18.8          ,67             ,-18.75         ,67.0666666667  ],
[-18.4833333333 ,67.1833333333  ,-18.3666666667 ,67.2333333333  ],
[-18.3666666667 ,67.0666666667  ,-18.35         ,67             ],
[-19.7333333333 ,67.3166666667  ,-19.8166666667 ,67.2666666667  ],
[-19.8166666667 ,67.15          ,-19.9333333333 ,67.1           ],
[-19.4          ,67.1666666667  ,-19.5333333333 ,67.1166666667  ],
[-20.7666666667 ,67.3           ,-20.6666666667 ,67.2333333333  ],
[-20.7666666667 ,67.1666666667  ,-20.9          ,67.1166666667  ],
[-20.6666666667 ,67.1           ,-20.8166666667 ,67.05          ],
[-20.55         ,67.3333333333  ,-20.7166666667 ,67.3           ],
[-20.15         ,67.0833333333  ,-20.2833333333 ,67.0333333333  ],
[-20.4666666667 ,67.15          ,-20.3          ,67.1666666667  ],
[-20.5666666667 ,67.0333333333  ,-20.45         ,67.0833333333  ],
[-21.45         ,67.0833333333  ,-21.6          ,67.0333333333  ],
[-21.2166666667 ,67.0333333333  ,-21.1666666667 ,67.1           ],
[-21.7166666667 ,67.1           ,-21.65         ,67.1666666667  ]])

intLines = np.array([ #first point is the one to add to path
[-21.21797313514038663, 65.9769674260938217 ,-22.45524153888934293 , 66.4484616330975939 ],
[-20.4926778639772067 , 66.07750828396038401, -20.24522418322741757, 65.6729554553719197 ],
[-20.13429667116716715, 66.16041487924540832, -19.61379065303829705, 65.72211978707083802],
[-19.51139602652114036, 66.10517392670715253, -19.37486985783160165, 65.75017176855935475],
[-18.78610075535796398, 66.21897661794660905, -18.1376014540826489 , 65.77819329403662607],
[-18.35092359266005602, 66.16386344331193925, -18.06080548419478404, 65.85160508785446609],
[-18.21439742397051731, 66.19832325644730986, -17.47203638172114637, 66.05673926993483747],
[-16.61874782741152146, 66.51655896001079782, -16.48222165872198275, 66.27053619081542024],
[-16.02997872493788023, 66.56072257371347689, -15.58626867669688032, 66.3151359445005113 ]])


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
            old = coords[x[j][k]]
            new = coords[x[j][k+1]]
            PI[j] += dist(old[0],old[1],old[2],old[3])
            PI[j] += dist(old[2],old[3],new[0],new[1])
        first = coords[x[j][0]]
        last = coords[x[j][od-1]]
        PI[j] += dist(last[0],last[1],last[2],last[3])
        PI[j] += dist(-18.18453232456968394,65.8830032158910228,first[0],first[1])
        PI[j] += dist(last[2],last[3],-14.98470024590859495,66.65726869659157217)
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

def reciprocalMutation(a):
    new = np.copy(a)
    i1 = np.random.randint(od)
    i2 = np.random.randint(od)
    new[i1], new[i2] = new[i2], new[i1]
    return new;

def dist(x1,y1,x2,y2):
    for i in intLines:
        if linesIntersect(i,[x1,y1,x2,y2]):
            return dist(x1,y1,i[0],i[1]) + dist(i[0],i[1],x2,y2)
            break
        else:
            if x1 > x2: # traveling west, apply 5% penalty in the x direction
                return np.sqrt( (((x2-x1)*1.05)**2) + (y2-y1)**2 )
            else:
                return np.sqrt( (x2-x1)**2 + (y2-y1)**2 )


def linesIntersect(l1,l2):
     a = np.array([l1[0], l1[1]])
     b = np.array([l1[2], l1[3]])
     c = np.array([l2[0], l2[1]])
     d = np.array([l2[2], l2[3]])
     if (np.cross(b-a,c-b) * np.cross(b-a,d-b) < 0) and (np.cross(d-c,a-d)*np.cross(d-c,b-d) < 0):
         return True
     else:
         return False

def orderCrossover(a1,a2):
    p1 = np.copy(a1)
    p2 = np.copy(a2)
    s1 = np.random.randint(0,od-1)
    s2 = np.random.randint(s1+1,od)
    child = np.empty(od) * np.nan
    child[s1:s2] = p2[s1:s2]
    toAdd = np.concatenate([p1[s2:],p1[:s2]])
    toAdd = np.delete(toAdd,np.where(np.in1d(toAdd,child)))
    avail = np.where(np.isnan(child))[0]
    child[avail] = toAdd
    return child

def linKernighan(array):
    a = np.copy(array)
    b = np.copy(array)
    r1 = np.random.randint(0,od-4)
    r2 = np.random.randint(r1+3,od-1)
    b[r1+1:r2+1] = a[r1+1:r2+1][::-1]
    x1 = dist(coords[a[r1]][2], coords[a[r1]][3], coords[a[r1+1]][0], coords[a[r1+1]][1])
    x2 = dist(coords[a[r2]][2], coords[a[r2]][3], coords[a[r2+1]][0], coords[a[r2+1]][1])
    y1 = dist(coords[b[r1]][2], coords[b[r1]][3], coords[b[r1+1]][0], coords[b[r1+1]][1])
    y2 = dist(coords[b[r2]][2], coords[b[r2]][3], coords[b[r2+1]][0], coords[b[r2+1]][1])
    if x1 + x2 > y1 + y2:
        a = np.copy(b)

    return a;



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
            #ng[i] = orderCrossover(x[p1],x[p2])
        elif PI[p1]<PI[p2]:
            ng[i] = x[p1]
        else:
            ng[i] = x[p2]
        #This is a quick fix for the edgerecombo missing the last element sometimes
        if np.unique(ng[i]).size != od:
            for j in range(0,od):
                if j not in ng[i]:
                    index = np.amax(np.where(ng[i]==od-1)[0]) # finds the last instance of od-1
                    ng[i][index] = j

    ### RECIPROCAL MUTATION
    vals = np.random.rand(1,npop)[0]
    toMut = np.where(vals<prmut)[0]
    for i in range(0,toMut.size):
        ng[i] = reciprocalMutation(ng[i])

    ### LIN KERNIGHAN
    for i in range(0,npop):
        array = np.copy(ng[i])
        ng[i] = np.copy(linKernighan(array))


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

### PRINT TO TXT FILE
#   prints the PI, array of points, and LINESTRING
#   LINESTRING is in WKT format for gis software
#   will be named PI.txt (205.94382398.txt)
bestPI = np.amin(endPI)
best = endX[np.argmin(endPI)]
linestring = "\nLINESTRING(-18.18453232456968394 65.88300321589102282,"
for h in range(0,od):
    linestring += str(coords[best[h]][0]) + " " + str(coords[best[h]][1]) + "," + str(coords[best[h]][2]) + " " + str(coords[best[h]][3])+","
linestring = linestring +'-14.98470024590859495 66.65726869659157217)\n'
f1=open('./'+str(bestPI)+'.txt', 'a+')
f1.write(str(bestPI))
f1.write("\n"+str(best))
f1.write(linestring)
f1.close()

### PLOT PI VALUES OVER GENERATIONS
#   will be named PI.png (205.94382398.png)
plt.plot(np.arange(1,endPI.size+1),endPI)
plt.ylabel('Performance Index')
plt.xlabel('Generation')
plt.xticks(np.arange(1,endPI.size+1,ngen/10))
plt.savefig('./'+str(bestPI)+'.png')




### TODOS
# Comment
# lin kernighan heuristic
