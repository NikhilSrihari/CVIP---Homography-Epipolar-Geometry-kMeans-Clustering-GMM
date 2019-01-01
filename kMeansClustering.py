import numpy as np
from math import sqrt
import matplotlib.pyplot as mplt
import matplotlib.markers as mmarkers


class KMeansClustering():

	def __init__(self, numOfClusters):
		self.K = numOfClusters
		self.dataPoints = None
		self.initialClusters = None
		self.numberOfDataPoints = None
		self.dataPointsDimension = None
		self.allClusterCenters = []
		self.finalClusterCenters = None
		self.allDataPointsDistribution = []
		self.finalDataPointsDistribution = None
		

	def __plotImage(self, imageType, iterNum):
		currentClusterCenters = (self.allClusterCenters[len(self.allClusterCenters)-1]).copy()
		currentClusterCenters = np.around(currentClusterCenters, decimals=2)
		currentDataPointsDistribution = self.allDataPointsDistribution[len(self.allDataPointsDistribution)-1]
		dp_x_coordinates = self.dataPoints[:,0]
		dp_y_coordinates = self.dataPoints[:,1]
		cc_x_coordinates = currentClusterCenters[:,0]
		cc_y_coordinates = currentClusterCenters[:,1]
		clusterGroupings = np.array([1, 2, 3])
		fig, ax = mplt.subplots()
		if (imageType=="Classification"):
			plotName = 'task3_iter'+str(iterNum)+'_a.jpg'
			dataGroupings = []
			i=0
			while(i<self.numberOfDataPoints):
				if (currentDataPointsDistribution[i][0]==1):
					dataGroupings.append(1)
				elif (currentDataPointsDistribution[i][1]==1):
					dataGroupings.append(2)
				else:
					dataGroupings.append(3)
				i=i+1
			dataGroupings = np.array(dataGroupings)
			ax.scatter(dp_x_coordinates[dataGroupings==1],dp_y_coordinates[dataGroupings==1], c='red', facecolors='full', marker=mmarkers.MarkerStyle(marker='^', fillstyle='full'), edgecolors='black')
			ax.scatter(dp_x_coordinates[dataGroupings==2],dp_y_coordinates[dataGroupings==2], c='green', facecolors='full', marker=mmarkers.MarkerStyle(marker='^', fillstyle='full'), edgecolors='black')
			ax.scatter(dp_x_coordinates[dataGroupings==3],dp_y_coordinates[dataGroupings==3], c='blue', facecolors='full', marker=mmarkers.MarkerStyle(marker='^', fillstyle='full'), edgecolors='black')
			ax.scatter(cc_x_coordinates[clusterGroupings==1],cc_y_coordinates[clusterGroupings==1], c='red', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='full'), edgecolors='red')
			ax.scatter(cc_x_coordinates[clusterGroupings==2],cc_y_coordinates[clusterGroupings==2], c='green', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='full'), edgecolors='green')
			ax.scatter(cc_x_coordinates[clusterGroupings==3],cc_y_coordinates[clusterGroupings==3], c='blue', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='full'), edgecolors='blue')
		else:
			plotName = 'task3_iter'+str(iterNum)+'_b.jpg'
			ax.scatter(dp_x_coordinates,dp_y_coordinates, facecolors='none', marker=mmarkers.MarkerStyle(marker='^', fillstyle='none'), edgecolors='black')
			ax.scatter(cc_x_coordinates[clusterGroupings==1],cc_y_coordinates[clusterGroupings==1], c='red', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='red')
			ax.scatter(cc_x_coordinates[clusterGroupings==2],cc_y_coordinates[clusterGroupings==2], c='green', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='green')
			ax.scatter(cc_x_coordinates[clusterGroupings==3],cc_y_coordinates[clusterGroupings==3], c='blue', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='blue')
		i=0
		while(i<self.numberOfDataPoints):
			ax.annotate("("+((str)(dp_x_coordinates[i]))+","+((str)(dp_y_coordinates[i]))+")", (dp_x_coordinates[i], dp_y_coordinates[i]))
			i=i+1
		i=0
		while(i<3):
			ax.annotate("("+((str)(cc_x_coordinates[i]))+","+((str)(cc_y_coordinates[i]))+")", (cc_x_coordinates[i], cc_y_coordinates[i]))
			i=i+1
		#mplt.show()
		fig.savefig(plotName, dpi=fig.dpi)


	def __calculateEuclideanDistance(self, x1, x2):
		if ((str(type(x1))=="<class 'numpy.ndarray'>") and (str(type(x2))=="<class 'numpy.ndarray'>")):
			x1 = np.array(x1)
			x2 = np.array(x2)
			return sqrt(np.sum(np.square(x1-x2)))
		else:
			return sqrt(x1**2 - x2**2)


	def fit(self, dataPoints, initialClusters=None, plotGraphs=False):
		self.dataPoints = dataPoints
		self.initialClusters = initialClusters if (str(type(initialClusters))=="<class 'numpy.ndarray'>") else self.dataPoints[0: self.K]
		self.numberOfDataPoints = (dataPoints).shape[0]
		self.dataPointsDimension = (dataPoints).shape[1]
		self.allClusterCenters.append(self.initialClusters)
		#Starting k means clustering logic
		iterNum=0
		while( (iterNum==0) or ((np.array_equal(currentClusterCenter, nextClusterCenter))==False) ):
			# For current cluster centers, find optimal point distribution
			currentClusterCenter = self.allClusterCenters[len(self.allClusterCenters)-1]
			currentDataPointsDistribution = np.zeros((self.numberOfDataPoints, self.K))
			i=0
			while(i<self.numberOfDataPoints):
				dataPoint = self.dataPoints[i]
				temp=[]
				j=0
				while(j<self.K):
					temp.append(self.__calculateEuclideanDistance(dataPoint, currentClusterCenter[j]))
					j=j+1
				currentDataPointsDistribution[i][temp.index(min(temp))] = 1 
				i=i+1
			(self.allDataPointsDistribution).append(currentDataPointsDistribution)
			if (plotGraphs==True):
				self.__plotImage("Classification", iterNum+1)
			# For current points distribution, find optimal cluster centers
			nume = (np.dot((np.transpose(currentDataPointsDistribution)), dataPoints))
			partial_den = (np.transpose(currentDataPointsDistribution)).sum(axis=1)
			den_create = []
			for a in range(self.dataPointsDimension):
				den_create.append(partial_den)
			den = np.transpose(np.array(den_create))
			nextClusterCenter = ( nume / den )
			self.allClusterCenters.append(nextClusterCenter)
			if (plotGraphs==True):
				self.__plotImage("UpdateClusterCenter", iterNum+1)
			# Increment iteration number
			iterNum=iterNum+1
			if ((self.numberOfDataPoints)>1000 and len(self.allClusterCenters)>=7):
				self.allClusterCenters = self.allClusterCenters[4:]
				self.allDataPointsDistribution = self.allDataPointsDistribution[4:]
		self.allClusterCenters = self.allClusterCenters[:len(self.allClusterCenters)-1]
		self.finalClusterCenters = nextClusterCenter
		self.finalDataPointsDistribution = currentDataPointsDistribution