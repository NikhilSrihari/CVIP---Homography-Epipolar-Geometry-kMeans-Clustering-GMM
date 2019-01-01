import numpy as np
import math 
import matplotlib.pyplot as mplt
import matplotlib.markers as mmarkers
from matplotlib.patches import Ellipse


class GaussianMixtureModel():

	def __init__(self, numOfClusters, taskNum):
		self.taskNum = taskNum
		self.C = numOfClusters
		self.Theta = None
		self.ProbabilityDistribution = None
		self.totNumOfIterations = None


	def __plot_cov_ellipse(self, cov, pos, nstd=2, ax=None, **kwargs):
		def eigsorted(cov):
			vals, vecs = np.linalg.eigh(cov)
			order = vals.argsort()[::-1]
			return vals[order], vecs[:,order]
		if ax is None:
			ax = mplt.gca()
		vals, vecs = eigsorted(cov)
		theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
		width, height = 2 * nstd * np.sqrt(vals)
		ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
		ax.add_artist(ellip)
		return ellip


	def __drawGraph(self, X, currTheta, p, N, iterNum):
		points = []
		j=0
		while(j<self.C):
			points.append([])
			j=j+1
		i=0
		while(i<N):
			(points[np.argmax(p[i])]).append(X[i])
			i=i+1
		fig = mplt.figure(0)
		ax = fig.add_subplot(111, aspect='equal')
		self.__plot_cov_ellipse(cov=currTheta["BigSigma"][0], pos=currTheta["Mu"][0], ax=ax, color='red', alpha=0.5)
		self.__plot_cov_ellipse(cov=currTheta["BigSigma"][1], pos=currTheta["Mu"][1], ax=ax, color='green', alpha=0.5)
		self.__plot_cov_ellipse(cov=currTheta["BigSigma"][2], pos=currTheta["Mu"][2], ax=ax, color='blue', alpha=0.5)
		ax.scatter((np.array(points[0]))[:,0],(np.array(points[0]))[:,1], c='red', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='red')
		ax.scatter((np.array(points[1]))[:,0],(np.array(points[1]))[:,1], c='green', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='green')
		ax.scatter((np.array(points[2]))[:,0],(np.array(points[2]))[:,1], c='blue', facecolors='full', marker=mmarkers.MarkerStyle(marker='o', fillstyle='none'), edgecolors='blue')
		ax.set_xlim(0, 110)
		ax.set_ylim(30, 110)
		fig.savefig("task3_gmm_iter"+(str)(iterNum+1)+".jpg", dpi=fig.dpi)


	def __BigSigmaValCheck(self, BigSigma, D):
		temp = BigSigma[0][0]
		i=0
		while(i<D):
			j=0
			while(j<D):
				if ( (i!=0 and j!=0) ):
					if (BigSigma[i][j]!=temp):
						return BigSigma
				j=j+1
			i=i+1
		BigSigma[0][0] = 1.1 * BigSigma[0][0]
		BigSigma[D-1][D-1] = 1.1 * BigSigma[D-1][D-1]
		return BigSigma
		

	def __calculateGaussianProb(self, x, Mu, BigSigma, D):
		BigSigma_i = np.linalg.inv(BigSigma)
		diff = x-Mu
		diff_t = np.transpose(diff)
		num = math.exp( -0.5 * (np.dot(np.dot(diff_t, BigSigma_i), diff)) )
		den = math.pow((2*3.14), (D/2)) * math.sqrt(np.linalg.det(BigSigma))
		return num/den


	def __logLikelihood(self, X, N, C, D, pi, Mu, BigSigma):
		tot = 0
		i=0
		while(i<N):
			x = X[i]
			summ = 0
			j=0
			while(j<self.C):
				summ = summ + ( pi[j] * self.__calculateGaussianProb(x, Mu[j], BigSigma[j], D) )
				j=j+1
			tot = tot + math.log(summ)
			i=i+1
		return tot


	def fit(self, X, initialParams):
		N = len(X)
		D = len(X[0])
		logLikelihoods = [-1000000000000000000]
		iterNum = 0
		while(1==1):
			#Step 1 : E
			if (iterNum==0):
				currTheta = initialParams
			else:
				currTheta = nextTheta
			currGuassProbMatrix = np.zeros((N,self.C))
			p = np.zeros((N,self.C))
			i=0
			while(i<N):
				x = X[i]
				j=0
				while(j<self.C):
					currGuassProbMatrix[i][j] = self.__calculateGaussianProb(x, (currTheta["Mu"][j]), (currTheta["BigSigma"][j]),  D)
					j=j+1
				i=i+1
			i=0
			while(i<N):
				j=0
				while(j<self.C):
					p[i][j] = ((currTheta["pi"][j])*(currGuassProbMatrix[i][j])) / (np.sum(currGuassProbMatrix[i]))
					j=j+1
				i=i+1
			#Step 2 : M
			m = np.sum(p, axis=0)
			m_sum = np.sum(m)
			pt_X = (np.dot(np.transpose(p), X))
			nextTheta = currTheta.copy()
			j=0
			while(j<self.C):
				nextTheta["pi"][j] = m[j] / m_sum
				j=j+1
			j=0
			while(j<self.C):
				nextTheta["Mu"][j] = (1/m[j]) * pt_X[j]
				j=j+1
			j=0
			while(j<self.C):
				summ = 0
				i=0
				while(i<N):
					diff = (X[i]-nextTheta["Mu"][j]).reshape(1, D)
					diff_t = np.transpose(diff)
					summ = summ + (p[i][j]) * np.dot( diff_t, diff )
					i=i+1
				nextTheta["BigSigma"][j] = self.__BigSigmaValCheck( ((1/m[j])*summ), D )
				j=j+1
			#Increment iterNum
			logLikelihood = self.__logLikelihood(X, N, self.C, D, nextTheta["pi"], nextTheta["Mu"], nextTheta["BigSigma"])
			if (self.taskNum=="A"):
				if (iterNum==0):
					print("		New Theta Values : "+str(nextTheta))
					print("		Current Theta Values : "+str(currTheta))
					print("		ProbabilityDistribution Matrix : "+str(p))
					print()
			else:
				if (iterNum>=0 and iterNum<=4):
					print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
					print("		New Theta Values : "+str(nextTheta))
					print("		Current Theta Values : "+str(currTheta))
					print("		ProbabilityDistribution Matrix : "+str(p))
					print()
					self.__drawGraph(X, currTheta, p, N, iterNum)
			if (logLikelihood<logLikelihoods[len(logLikelihoods)-1]):
				self.Theta =currTheta
				self.ProbabilityDistribution = p
				self.totNumOfIterations = iterNum
				self.finalLogLikelihood = logLikelihoods[len(logLikelihoods)-1]
				break
			else:
				if (iterNum>100):
					print("		Breaking coz there was no convergence in 100 iterations")
					break
				else:
					logLikelihoods.append(logLikelihood)
			iterNum = iterNum + 1