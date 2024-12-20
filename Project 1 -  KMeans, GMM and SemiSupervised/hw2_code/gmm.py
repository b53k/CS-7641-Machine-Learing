import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """

        #raise NotImplementedError
        # To ensure numeric stability
        max_each_row = np.max(logit, axis = 1).reshape(-1,1)
        logit = logit - max_each_row
        prob = np.exp(logit)/(np.sum(np.exp(logit), axis = 1)).reshape(-1,1)
        return prob

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """

        #raise NotImplementedError
        val = np.max(logit,axis=1).reshape(-1,1)
        logit = logit - val
        return (np.log(np.sum(np.exp(logit),axis=1))+ val.squeeze()).reshape(logit.shape[0],1)

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        #raise NotImplementedError
        try:
            D = np.size(points[1])
        except:
            D = np.size(points[0])

        frac = 1/((2*np.pi)**(D/2))
        x_mu = points - mu_i.reshape(1,-1)

        try:
            #print (sigma_i)
            det = np.linalg.det(sigma_i)**(-0.5)
        except:
            det = np.linalg.det(sigma_i + SIGMA_CONST)**(-0.5) #1e-32
        
        try:
            sigma_inv = np.linalg.inv(sigma_i)
        except:
            sigma_inv = np.linalg.inv(sigma_i + SIGMA_CONST) #1e-32

        # calculate (x-mu)*sigma_inverse and take transpose
        xmsi = np.transpose(np.matmul(x_mu,sigma_inv))
        # element wise multiply above with (x-mu).T
        inner = np.multiply(xmsi,np.transpose(x_mu))
        # now sum inner aong axis = 0
        inner = -0.5*np.sum(inner, axis = 0)

        exp = np.exp(inner)
        normal_pdf = frac*det*exp


        return normal_pdf.reshape(-1)



    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        np.random.seed(5) #Do not remove this line!
        #raise NotImplementedError
        pi = np.array([1 / self.K for i in range(self.K)])
        random_int = np.random.randint(low=0, high=self.points.shape[0], size=self.K)
        mu = np.array([self.points[i,:] for i in random_int])
        sigma = np.stack([np.identity(self.D) for j in range(self.K)], axis = 0)
        #print (sigma)
        #print (sigma.shape)
        

        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        #raise NotImplementedError
        
        inside_log = np.stack([self.multinormalPDF(self.points, mu[i, :], sigma[i, :]) for i in range(self.K)], axis = 1)
        #arr = []
        #for i in range(self.K):
        #    multi_normal = self.multinormalPDF(self.points, mu[i, :], sigma[i, :]).reshape(1,-1)
        #    arr.append(multi_normal)
        #normal = np.vstack(arr)
        #print (normal)
        result = np.log(inside_log + LOG_CONST) + np.log(pi)
        return result


    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...
        

        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        #raise NotImplementedError
        log_like = self._ll_joint(pi, mu, sigma) # need to pass it to ll_joint
        return self.softmax(log_like)
        

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...

        #raise NotImplementedError
        N, K = gamma.shape
        D = self.points.shape[1]

        sum = np.sum(gamma, axis = 0)
        pi = sum/(len(gamma))
        mu = np.dot(gamma.T, self.points)/sum.reshape(-1,1)
        sigma = np.zeros((K,D,D))

        for k in range(K):
            x_mu = self.points - mu[k]
            A = gamma[:, k].T * x_mu.T
            sigma[k] = np.dot(A, x_mu)/np.sum(gamma[:,k], axis = 0)

        return pi, mu, sigma



    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)