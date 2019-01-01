import numpy as np
from kMeansClustering import KMeansClustering
import cv2


baboonImageLocation = './proj2_data/data/baboon.jpg'


def readImage(imageLocation):
    img = cv2.imread(imageLocation, 1)
    return img


def writeImage(img, outputFileName):
	cv2.imwrite(outputFileName, img)
	return 1


def main():
	print("Task 3 :")
	print("	Task 3.1, 3.2, 3.3 :")
	X = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]])
	initialClusters = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])
	kMeansCluster = KMeansClustering(3)
	kMeansCluster.fit(X, initialClusters, True)
	print("		All Cluster Centers : "+(str)(kMeansCluster.allClusterCenters))
	print("		Final Cluster Centers : "+(str)(kMeansCluster.finalClusterCenters))
	print("		All Data Points Distribution : "+(str)(kMeansCluster.allDataPointsDistribution))
	print("		Final Data Points Distribution : "+(str)(kMeansCluster.finalDataPointsDistribution))
	print()
	'''print("	Task 3.4 :")
	baboonImg = readImage(baboonImageLocation)
	#baboonImg = cv2.resize(baboonImg , (128, 128))
	baboonData = []
	i=0
	while(i<len(baboonImg)):
		j=0
		while(j<len(baboonImg[0])):
			baboonData.append(baboonImg[i][j])
			j=j+1
		i=i+1
	baboonData = np.array(baboonData)
	K = [3, 5, 10, 20]
	for k in K:
		baboonImg_copy = baboonImg.copy()
		kMeansCluster = KMeansClustering(k)
		kMeansCluster.fit(baboonData)
		print("		Final Cluster Centers for K = "+(str)(k)+" : "+(str)(kMeansCluster.finalClusterCenters))
		i=0
		while(i<len(baboonImg)):
			j=0
			while(j<len(baboonImg[0])):
				baboonImg_copy[i][j] = kMeansCluster.finalClusterCenters[np.argmax(kMeansCluster.finalDataPointsDistribution[(i*len(baboonImg))+j])]
				j=j+1
			i=i+1
		writeImage(baboonImg_copy, "task3_baboon_"+str(k)+".jpg")'''
	

main()