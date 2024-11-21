'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np
#import random

class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [2 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        #raise NotImplementedError
        #instance_id = random.sample(range(0, points.shape[0]), k = K)
        #centers = points[instance_id, :]
        '''Couldn't use above because Autograder won't let me import random'''
        centers = points[np.random.choice(points.shape[0], K, replace = False)]
        return centers

    def _kmpp_init(self, points, K, **kwargs): # [3 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        
        
        #centroids = np.zeros((K,points.shape[1]))
        '''centroids = []
        one_percent = len(points)//100
        sample = points[np.random.choice(len(points), one_percent)]
        centroids = [sample[np.random.randint(sample.shape[0]),:]]
        print (centroids)'''

        raise NotImplementedError


    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        
        #raise NotImplementedError
        #matrix = pairwise_dist(points, centers)   # N x K
        #cluster = np.where(np.amin(matrix, axis = 1))
        #cluster_idx = np.array(cluster)
        #return cluster_idx.squeeze()
        matrix = pairwise_dist(points, centers)
        cluster_idx = np.argmin(matrix, axis = 1)
        return cluster_idx


    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        #raise NotImplementedError
        centers = []
        for k in range(old_centers.shape[0]):
            cluster_k = points[np.argwhere(cluster_idx==k)][:,0,:]
            new_cluster = np.mean(cluster_k, axis=0) 
            centers.append(new_cluster)
        return np.asarray(centers).reshape(old_centers.shape)



    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """
        #raise NotImplementedError
        #loss = 0
        #for k in range(centers.shape[0]):
        #    loss += np.square(np.linalg.norm(points[cluster_idx == k] - centers[k])) # Euclidean norm squared
        #return loss
        #loss = 0
        #for k in range()
        loss = 0
        for i, center_k in enumerate(centers):
            cluster_k = points[np.argwhere(cluster_idx == i)]
            hold = np.sum(np.square(pairwise_dist(cluster_k, center_k)), axis = 1).sum()
            loss += hold
        return loss


    def _get_centers_mapping(self, points, cluster_idx, centers):
        # This function has been implemented for you, no change needed.
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        This function has been implemented for you, no change needed.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss


def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    """
    #raise NotImplementedError
    #x = x.reshape(x.shape[0],1,-1)
    #y = y.reshape(1,y.shape[0],-1)
    #dist = np.linalg.norm(x-y, axis = -1)
    #return dist
    x = x[:,:,None]
    y = y.T
    return np.sqrt(np.sum(np.square(y-x), axis=1))

def silhouette_coefficient(points, cluster_idx, centers, centers_mapping): # [10pts]
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """
    #raise NotImplementedError
    mu_ins = np.zeros([points.shape[0],1])
    mu_outs = np.zeros([points.shape[0],1])
    si = 0
    a = np.unique(cluster_idx)

    for i in range(len(points)):
        pt_i = points[i]
        #cw_pt_i = centers_mapping.get(i) #>>>
        cw_pt_i = points[cluster_idx == cluster_idx[i]]
        clone_pt_i = np.tile(pt_i, (cw_pt_i.shape[0], 1))
        mu_i = np.sqrt(np.sum(np.square(clone_pt_i - cw_pt_i), axis = 1))
        mu_i = np.sum(mu_i)/(len(cw_pt_i)-1)
        mu_ins[i] = mu_i

        current_cluster = cluster_idx[i]
        hold = []
        for j in range(len(a)):
            if current_cluster != a[j]:
                points_j = points[cluster_idx == a[j]]
                clone_cp = np.tile(pt_i, (points_j.shape[0], 1))
                mu_ji = np.sqrt(np.sum(np.square(clone_cp - points_j), axis = 1))
                mu_ji = np.sum(mu_ji)/(len(points_j))
                hold.append(mu_ji)
        mu_outs[i] = np.asarray(hold).min()

        si += (np.asarray(hold).min() - mu_i)/max(np.asarray(hold).min(), mu_i)
    
    return si/points.shape[0], mu_ins, mu_outs