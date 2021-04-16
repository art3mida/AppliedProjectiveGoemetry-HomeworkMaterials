import sys
import math
import numpy as np
import numpy.linalg as LA

def R_x(angle):
    return np.array([
        [1,               0,                0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle),  math.cos(angle)]
    ])

def R_y(angle):
    return np.array([
        [ math.cos(angle), 0, math.sin(angle)],
        [               0, 1,               0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])

def R_z(angle):
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [              0,                0, 1]
    ])

def euler2a(phi, theta, psi):
    Rx = R_x(phi)
    Ry = R_y(theta)
    Rz = R_z(psi)
    return (Rz @ Ry) @ Rx

def normalize(v):
    return v/math.sqrt(sum([x**2 for x in v]))

def check_matrix(A):
    # Ovim kodom proveravamo da li je matrica ortogonalna i da li je njena determinanta 1 (zbog
    # racuna sa decimalnim brojevima, postoji mogucnost greske, pa proveravamo da li je 
    # "dovoljno blizu" 1, tj. da li je razlika dovoljno blizu 0).
    if abs(LA.det(A)-1) >= 0.00001 or ((A.T @ A) != np.eye(3)).all():
        print('Matrica nije validna!')
        return False
    return True

def axis_angle(A):
    # Prvo je potrebno proveriti da li je matrica ortogonalna i da li je njena determinanta 1.
    # Ako nije, znamo da se ne radi o rotaciji.
    if not check_matrix(A):
        print('Matrica nije validna.')
        return
    
    B = A - np.eye(3)
    p = np.cross(B[0], B[1])
    if not np.any(p):
        p = np.cross(B[0], B[2])
        if not np.any(p):
            p = np.cross(B[1], B[2])
    p = normalize(p)

    x = B[0]
    if not np.any(x):
        x = B[1]
        if not np.any(x):
            x = B[2]
    x = normalize(x)
    xp = np.matmul(A, x)

    angle = math.acos(np.dot(x, xp))
    if LA.det(np.array([x, xp, p])) < 0:
        p = -p
    
    return (p, angle)

def rodrigez(p, angle):
    p = normalize(p)
    ppt = p.reshape(3, -1) * p
    px = np.array([
        [0, -p[2], p[1]], 
        [p[2], 0, -p[0]], 
        [-p[1], p[0], 0]
    ])
    R = ppt + np.cos(angle)*(np.eye(3)-ppt) + np.sin(angle)*px
    return R

def a2euler(A):
    # Opet moramo proveriti da li je matrica rotacija.
    if not check_matrix(A):
        return

    if A[2, 0] < 1:
        if A[2, 0] > -1:
            psi = math.atan2(A[1,0], A[0,0])
            theta = math.asin(-A[2,0])
            phi = math.atan2(A[2,1], A[2,2])
        else:
            psi = math.atan2(-A[0,1], A[1,1])
            theta = math.pi/2
            phi = 0
    else:
        psi = math.atan2(-A[0,1], A[1,1])
        theta = -math.pi/2
        phi = 0
    
    return (phi, theta, psi)

def axisangle2q(p, angle):
    w = math.cos(angle/2)
    p = normalize(p)
    x, y, z = math.sin(angle/2)*p
    return np.array([x, y, z, w])

def q2axisangle(q):
    q = normalize(q)
    if q[3] < 0:
        q = -1 * q
    angle = 2 * math.acos(q[3])
    if abs(q[3]) == 1:
        p = np.array([1, 0, 0])
    else:
        p = np.array(q[0:3])
        p = normalize(p)
    return (p, angle)

def slerp(q1, q2, tm, t):
    if(t == 0):
        return q1
    elif(t == tm):
        return q2

    sk = np.dot(q1,q2) / np.linalg.norm(q1) * np.linalg.norm(q2)

    if(sk < 0):
        q1 = -q1
        sk = -sk

    phi = math.acos(sk)
    qs = (math.sin (phi * (1 - t/tm)) / math.sin(phi)) * np.array(q1) + (math.sin(phi * t/tm) / math.sin(phi)) * np.array(q2)

    return qs