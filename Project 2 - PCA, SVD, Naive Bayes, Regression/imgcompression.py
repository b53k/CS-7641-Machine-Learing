import numpy as np
from typing import Tuple


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images ((N,D) arrays) as well as color images ((N,D,3) arrays)
        In the image compression, we assume that each column of the image is a feature. Perform SVD on the channels of
        each image (1 channel for black and white and 3 channels for RGB)
        Image is the matrix X.

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images

        Return:
            U: (N,N) numpy array for black and white images / (N,N,3) numpy array for color images
            S: (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V^T: (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
        """
        #raise NotImplementedError
        N, D = X.shape[0], X.shape[1]

        if X.ndim == 2:
            U,S,Vh = np.linalg.svd(X, full_matrices = True)

        else:
            U = np.zeros((N,N,3))
            S = np.zeros((np.minimum(N,D), 3))
            Vh = np.zeros((D,D,3))

            #r = X[:,:,0]
            #g = X[:,:,1]
            #b = X[:,:,2]
            #ur, sr, vr = np.linalg.svd(r, full_matrices = True)
            #ug, sg, vg = np.linalg.svd(g, full_matrices = True)
            #ub, sb, vb = np.linalg.svd(b, full_matrices = True)
            for idx in range(3):
                u, s, v = np.linalg.svd(X[:,:,idx], full_matrices = True)
                U[:,:,idx] = u
                S[:, idx] = s
                Vh[:,:,idx] = v

        return U,S,Vh


    def compress(
        self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # [4pts]
        """Compress the SVD factorization by keeping only the first k components

        Args:
            U (np.ndarray): (N,N) numpy array for black and white simages / (N,N,3) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k (int): int corresponding to number of components to keep

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                U_compressed: (N, k) numpy array for black and white images / (N, k, 3) numpy array for color images
                S_compressed: (k, ) numpy array for black and white images / (k, 3) numpy array for color images
                V_compressed: (k, D) numpy array for black and white images / (k, D, 3) numpy array for color images
        """
        #raise NotImplementedError
        if U.ndim == 2:
            U_compressed = U[:,:k]
            S_compressed = S[:k]
            V_compressed = V[:k,:]
        else:
            U_compressed = U[:,:k,:]
            S_compressed = S[:k,:]
            V_compressed = V[:k,:,:]
        
        return U_compressed, S_compressed, V_compressed


    def rebuild_svd(
        self,
        U_compressed: np.ndarray,
        S_compressed: np.ndarray,
        V_compressed: np.ndarray,
    ) -> np.ndarray:  # [4pts]
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.

        Args:
            U_compressed: (N,k) numpy array for black and white images / (N,k,3) numpy array for color images
            S_compressed: (k, ) numpy array for black and white images / (k,3) numpy array for color images
            V_compressed: (k,D) numpy array for black and white images / (k,D,3) numpy array for color images

        Return:
            Xrebuild: (N,D) numpy array of reconstructed image / (N,D,3) numpy array for color images

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        #raise NotImplementedError
        N, D = U_compressed.shape[0], V_compressed.shape[1]

        if U_compressed.ndim == 2:
            diag = np.diag(S_compressed)
            Xrebuild = np.matmul(np.matmul(U_compressed, diag),V_compressed)
        else:
            Xrebuild = np.zeros((N, D, 3))
            for i in range(3):
                diag = np.diag(S_compressed[:,i])
                Xrebuild[:,:,i] = np.matmul(np.matmul(U_compressed[:,:,i], diag), V_compressed[:,:,i])
        
        return Xrebuild


    def compression_ratio(self, X: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the compression ratio of an image: (num stored values in compressed)/(num stored values in original)

        Args:
            X: (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        #raise NotImplementedError

        N, D = X.shape[0], X.shape[1]
        num = k*(N + D + 1)
        denum = N*D

        compression_ratio = num/denum
        
        return compression_ratio


    def recovered_variance_proportion(self, S: np.ndarray, k: int) -> float:  # [4pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (min(N,D), ) numpy array black and white images / (min(N,D),3) numpy array for color images
           k: int, rank of approximation

        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        #raise NotImplementedError
        #S = np.asarray(S)
        if S.ndim == 1:
            num = np.sum(np.square(S[:k]))
            denum = np.sum(np.square(S))
            recovered_variance = num/denum
        else:
            recovered_variance = []
            for i in range(S.shape[1]):
                num = np.sum(np.square(S[:k, i]))
                denum = np.sum(np.square(S[:, i]))
                recovered_variance.append(num/denum)
    
        return np.asarray(recovered_variance)


    def memory_savings(
        self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS
        
        Returns the memory required to store the original image X and 
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (N,D) numpy array corresponding to black and white images / (N,D,3) numpy array for color images
            U (np.ndarray): (N,N) numpy array for black and white simages / (N,N,3) numpy array for color images
            S (np.ndarray): (min(N,D), ) numpy array for black and white images / (min(N,D),3) numpy array for color images
            V (np.ndarray): (D,D) numpy array for black and white images / (D,D,3) numpy array for color images
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]: 
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X 
        """

        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes

        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"