import numpy as np
from numpy import linalg as la
import math

def naive_algorithm_canon(points):
    A = points[0]
    B = points[1]
    C = points[2]
    D = points[3]
    
    A0 = (1, 0, 0)
    B0 = (0, 1, 0)
    C0 = (0, 0, 1)
    D0 = (1, 1, 1)
    
    delta = [ [A[0], B[0], C[0]],
              [A[1], B[1], C[1]],
              [A[2], B[2], C[2]] ]
    
    delta1 = [ [D[0], B[0], C[0]],
               [D[1], B[1], C[1]],
               [D[2], B[2], C[2]] ]
    
    
    delta2 = [ [A[0], D[0], C[0]],
               [A[1], D[1], C[1]],
               [A[2], D[2], C[2]] ]
    
    delta3 = [ [A[0], B[0], D[0]],
               [A[1], B[1], D[1]],
               [A[2], B[2], D[2]] ]
    
    delta_det = la.det(delta)
    delta1_det = la.det(delta1)
    delta2_det = la.det(delta2)
    delta3_det = la.det(delta3)
    
    lambda1 = delta1_det/delta_det
    lambda2 = delta2_det/delta_det
    lambda3 = delta3_det/delta_det
    
    
    P = [[lambda1*x for x in A],
         [lambda2*x for x in B],
         [lambda3*x for x in C]]
    
    
    return np.transpose(P)

def naive_algorithm(points, points_proj):
    P1 = naive_algorithm_canon(points)
    P2 = naive_algorithm_canon(points_proj)

    P1_inv = la.inv(P1)
    P = np.matmul(P2, P1_inv)
    
    return P

def big_matrix(t1, t2):
    M = np.matrix(
        [ [0, 0, 0, -t2[2]*t1[0], -t2[2]*t1[1], -t2[2]*t1[2], t2[1]*t1[0], t2[1]*t1[1], t2[1]*t1[2]],
         [t2[2]*t1[0], t2[2]*t1[1], t2[2]*t1[2], 0, 0, 0, -t2[0]*t1[0], -t2[0]*t1[1], -t2[0]*t1[2]]
        ])
    return M

def dlt_algorithm(points, points_proj):
    A = []
    n = len(points)
    for i in range(n):
        a = big_matrix(points[i], points_proj[i])
        
        if(i > 0):
            A = np.concatenate((A, a), axis = 0)
        else:
            A = a
    
    U, D, Vt = la.svd(A, full_matrices=True)
    V = np.transpose(Vt)
    V = V[:, -1]
    P = V.reshape(3,3)
    
    return P

def normalize(points):
    # CMS - center of mass (teziste)
    CMS = []
    affine_points = []
    n = len(points)
    for i in range(n):
        A = np.array(points[i])
        A = np.array([A[0]/A[2], A[1]/A[2]])
        affine_points.append(A)
        if(i > 0):
            CMS += A
        else:
            CMS = A
    CMS = CMS * np.array([1/n, 1/n])
    
    # Translacija
    T = np.matrix([ [1, 0, -CMS[0]],
                    [0, 1, -CMS[1]],
                    [0, 0, 1] ])
    
    # Homotetija
    lamb = 0
    for i in range(n):
        A = affine_points[i] 
        lamb += math.sqrt(A[0]*A[0] + A[1]*A[1])
    lamb = lamb / n
    
    h = math.sqrt(2) / lamb
    H = np.matrix([[h, 0, 0],
                   [0, h, 0],
                   [0, 0, 1]
                   ])
    
    # Matrica normalizacije
    N = np.matmul(H, T)
    return N

def dlt_normalize(points, points_proj):    
    T_matrix = normalize(points) 
    Tp_matrix = normalize(points_proj) 
    
    T_matrix = np.array(T_matrix).reshape((3,3))
    Tp_matrix = np.array(Tp_matrix).reshape((3,3))
  
    points = np.transpose(points)
    points_proj = np.transpose(points_proj)
    
    M_nadvuceno = T_matrix.dot(points) 
    Mp_nadvuceno = Tp_matrix.dot(points_proj) 
    
    M_nadvuceno = np.transpose(M_nadvuceno) 
    Mp_nadvuceno = np.transpose(Mp_nadvuceno) 
    dlt_matrix = dlt_algorithm(M_nadvuceno, Mp_nadvuceno) 
    
    dlt_matrix = np.array(dlt_matrix).reshape((3, 3))

    result = (np.linalg.inv(Tp_matrix)).dot(dlt_matrix).dot(T_matrix)
    return result