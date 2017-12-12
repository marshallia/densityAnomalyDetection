import scipy.io.arff as sc
import scipy.spatial.distance as dst
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
1. cari distance for each node		(done)
2. cari sort ambil k node teratas	(done)
3. cari density(1/average(k node))	(done)
4. cari average lof (average(density node)/density node x)
'''
data,meta=sc.loadarff('anomaly.arff')
X=pd.DataFrame(data)
data=np.array(X.iloc[0:5])
plt.figure(1)
plt.plot(X['x'],X['y'],'bo',)
plt.xlabel('x')
plt.ylabel('y')
threshold=5
k=3
datatype=np.dtype([('jarak',np.float64),('nomor',np.int32)])
dist=np.zeros(shape=(5,5),dtype=datatype)#create matrix distance
dens=np.zeros(shape=(1,len(X)),dtype=float)
urutan=np.zeros(shape=(len(X),k),dtype=int)
LOF=list()
outliers=list()
def distance(u,v):
	dist=0;
	dist=dst.euclidean(u,v)
	return dist
def density(k,u):
	neigh=np.array(np.sort(u,order='jarak'))
	urut=[]
	sumDist=0	
	for i in range(k+1):
		sumDist+=neigh[i][0]
		if i!=0:		
			urut.append(neigh[i][1])
	print('density=1/average(distance)')	
	print('density = 1/','(',sumDist,'/',k,')')
	density=k/sumDist
	return density,urut
def aveDensity():
	for j in range(len(X)):
		sumDens=0
		for i in range(k):
			sumDens+=dens[0][urutan[j][i]]
		help=sumDens/k
		average=help/dens[0][j]		
		if average>=threshold:
			outliers.append(j)		
		print('LOF',j,'=',sumDens,'/',k,'/',dens[0][j])
		print('LOF',j,'= ',average)
		LOF.append(average)
for i in range(len(X)):
	for j in range(len(X)):
		dist[i][j]=distance(data[i],data[j]),j
	dens[0][i],urutan[i]=density(k,dist[i])
	print('urutan k neighbors terdekat ',urutan[i])
	print("density ", i , " ",dens[0][i])

aveDensity()
outX=list()
outY=list()
for i in range(len(outliers)):
	outX.append(X.loc[outliers[i],['x']])
	outY.append(X.loc[outliers[i],['y']])
print (outX,outY)
plt.figure(2)
plt.plot(X['x'],X['y'],'bo',outX,outY,'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.show()
