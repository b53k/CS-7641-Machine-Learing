import numpy as np
from kmeans import pairwise_dist

class DBSCAN(object):
    
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

        self.visited_indices = []
        self.visited_points = []
        
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        #raise NotImplementedError
        noise = []
        C = 0
        cluster_idx = []

        for index, point in enumerate(self.dataset):
            self.visited_indices.append(index) # marked visited index
            self.visited_points.append(point) # marked visited point

            neighbourPts = self.regionQuery(point) # indices (I,) numpy

            if len(neighbourPts) < self.minPts:
                noise.append(point)   # not index but actual points
            else:
                cluster_idx.append([])
                self.expandCluster(index, neighbourPts, C, np.asarray(cluster_idx), self.visited_indices)
                C += 1
        
        return np.array(cluster_idx)


    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, 
        and any points density-reachable to P and 
        updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all 
            points within P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment 
            of clusters for each point in dataset
            visitedIndices: set of indsamplingices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() 
            may be helpful here
            A while loop may be better than a for loop
        """

        #raise NotImplementedError
        Point = self.dataset[index]
        cluster_idx = cluster_idx.tolist()
        #print (cluster_idx[C])
        cluster_idx[C].append(Point)
        
        visited_points = self.dataset[np.asarray(visitedIndices)] # should work 
        neighbourPts = self.dataset[neighborIndices] # already numpy array...should work

        '''Note: VisitedIndices = self.visited_indices []'''

        for point in neighbourPts:

            if point not in visited_points:
                index = np.where((self.dataset == point).all(1))[0][0]
                self.visited_indices.append(index)
                new_neighbourPts = self.regionQuery(index)
                
                if len(new_neighbourPts) >= self.minPts:
                    neighbourPts = np.sort(np.concatenate((neighbourPts, new_neighbourPts)))
            
            if point not in (i for i in cluster_idx):
                cluster_idx[C].append(point)


        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        #raise NotImplementedError
        neighbors = []
        dist_matrix = pairwise_dist(self.dataset, self.dataset)
        for i in range(len(dist_matrix)):
            #if pointIndex != i:
            if dist_matrix[pointIndex][i] <= self.eps:
                neighbors.append(i)
        return np.asarray(neighbors)
