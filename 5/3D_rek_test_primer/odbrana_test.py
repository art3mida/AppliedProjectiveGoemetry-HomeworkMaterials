##########################################
##### Test primer #######################
#########################################

import numpy as np
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

# Piksel koordinate 8 vidljivih tacaka.
# x1 = np.array([958, 38, 1.])
# y1 = np.array([933, 33, 1.])
# x2 = np.array([1117, 111, 1.])
# y2 = np.array([1027, 132, 1.])
# x3 = np.array([874, 285, 1.])
# y3 = np.array([692, 223, 1.])
# x4 = np.array([707, 218, 1.])
# y4 = np.array([595, 123, 1.])
# x9 = np.array([292, 569, 1.])
# y9 = np.array([272, 360, 1.])
# x10 = np.array([770, 969, 1.])
# y10 = np.array([432, 814, 1.])
# x11 = np.array([770, 1465, 1.])
# y11 = np.array([414, 1284, 1.])
# x12 = np.array([317, 1057, 1.])
# y12 = np.array([258, 818, 1.])

# # Preostale vidljive tacke
# x6 = np.array([1094, 536, 1.])
# y6 = np.array([980, 535, 1.])
# x7 = np.array([862, 729, 1.])
# y7 = np.array([652, 638, 1.])
# x8 = np.array([710, 648, 1.])
# y8 = np.array([567, 532, 1.])
# x14 = np.array([1487, 598, 1.])
# y14 = np.array([1303, 700, 1.])
# x15 = np.array([1462, 1079, 1.])
# y15 = np.array([1257, 1165, 1.])
# y13 = np.array([1077, 269, 1.])


#leva:
x1 = np.array([1960, 650, 1.])
x2 = np.array([2305, 750, 1.])
x3 = np.array([1680, 1000, 1.])
x4 = np.array([1350, 895, 1.])
x9 = np.array([805, 1770, 1.])
x10 = np.array([1890, 2075, 1.])
x11 = np.array([1850, 2370, 1.])
x12 = np.array([825, 2000, 1.])

#desna:
y1 = np.array([2175, 430, 1.])
y2 = np.array([2445, 620, 1.])
y3 = np.array([1505, 815, 1.])
y4 = np.array([1310, 590, 1.])
y9 = np.array([760, 1400, 1.])
y10 = np.array([1470, 2010, 1.])
y11 = np.array([1450, 2265, 1.])
y12 = np.array([790, 1600, 1.])

#preostale leva:
x6 = np.array([2300, 1430, 1.])
# x6 = np.array[1300, 1430, 1.]
x7 = np.array([1740, 1755, 1.])
x8 = np.array([1420, 1590, 1.])
#x13-?
#x14 = np.array[2700, 1210, 1.]
x14 = np.array([2560, 1210, 1.])
x15 = np.array([2550, 1455, 1.])
#x16-?

#preostale desna:
y6 = np.array([2395, 1360, 1.])
y7 = np.array([1575, 1600, 1.])
y8 = np.array([1400, 1345, 1.])
# y13-?
y14 = np.array([2870, 1180, 1.])
y15 = np.array([2810, 1430, 1.])
#y-16?


# Vektori tih tacaka, koje cemo proslediti funkciji (bez normalizacije)
# Los izlaz
xx = np.array([x1, x2, x3, x4, x6, x7, x8, x9, x10, x11, x12, x14, x15])
yy = np.array([y1, y2, y3, y4, y6, y7, y8, y9, y10, y11, y12, y14, y15])

# Najbolji izlaz
# xx = np.array([x1, x2, x6, x7, x9, x10, x14, x15])
# yy = np.array([y1, y2, y6, y7, y9, y10, y14, y15])


# Normalizacija
def norm(vertex):
    return vertex/vertex[-1]

# Funkcija za proveru da li je y^T * F * x = 0 za 
# odgovarajuce tacke.
def test(x, y, FF):
    return y @ FF @ x

# Funkcija za nalazenje koordinata nevidljivih tacaka vektorskim proizvodom.
def findInvisible(a, b, c, d, e, f, g, h, i, j):
    vertex = np.cross(
            np.cross(np.cross(np.cross(a, b), np.cross(c, d)), e),
            np.cross(np.cross(np.cross(f, g), np.cross(h, i)), j))

    vertexNormed = np.round(norm(vertex))
    return vertexNormed

# Matrica vektorskog mnozenja
def vec(p):
    return np.array([[  0,   -p[2],  p[1]],
                    [ p[2],   0,   -p[0]],
                    [-p[1],  p[0],   0  ]])

def reconstruct():
    # Jednacina y^T * F * x = 0.
    # Nepoznate su koeficijelni fundamentalne matrice F.
    jed = lambda x, y: np.array([np.outer(y, x).flatten()])

    # Osam jednacina dobijenih iz korespondencija (8x9).
    jed8 = np.concatenate([jed(x, y) for x, y in zip(xx, yy)])
    print('jed8: ')
    print(jed8)

    # DLT algoritam.
    # Radimo SVD dekompoziciju matrice jednacina formata 8x9.
    SVDJed8 = LA.svd(jed8)

    # Koeficijenti marice su poslednja kolona matrice V.
    Fvector = SVDJed8[-1][-1]

    # Od toga pravimo matricu F i nazivamo je FF.
    FF = Fvector.reshape(3, 3)

    # Proveravamo da li je y^T * F * x = 0 za odgovarajuce tacke.
    # Vektor testrez treba imati elemente bliske nuli.
    testIfCloseToZero = np.array([test(x, y, FF) for x, y in zip(xx, yy)])
    print('Rezultati jednacine: ', testIfCloseToZero)

    # Determinanta fundamentalne matrice takodje treba biti bliska nuli.
    det = LA.det(FF)
    print('Determinanta FF: ', det)

    # Da bismo nasli epipol e1 treba da resimo sistem F * e1 = 0.
    # To mozemo da uradimo SVD dekompozicijom matrice FF.
    SVDFF = LA.svd(FF)
    U, DD, VT = SVDFF
    print('Matrica U:')
    print(U)
    print('Matrica DD:')
    print(DD)
    print('Matrica VT')
    print(VT)

    # Treca vrsta V^T je trazeni epipol.
    # Ta kolona odgovara najmanjoj sopstvenoj vrednosti matrice.
    e1 = VT[-1]
    print('e1: ', e1)

    # Afine koordinate epipola e1
    e1 = norm(e1)
    print('Afine koordinate epipola e1:', e1)

    # Za drugi epipol treba resiti F^T * e2 = 0, ali primecujemo da je
    # SVD dekompozicija F^T zapravo samo transponovana SVD dekompozicija
    # F, tako da je drugi epipol treca (poslednja) kolona matrice U iz prvog
    # razlaganja.
    e2 = U[:, -1]
    print('e2: ', e2)

    # Afine koordinate epipola e2
    e2 = norm(e2)
    print('Afine koordinate epipola e2: ', e2)

    #######################################
    ##### Postizanje uslova det(FF)=0 #####
    #######################################
    print('-----------------------------------------------')

    # Zeljena matrica je singularna
    DD1 = np.diag([1, 1, 0]) @ DD
    DD1 = np.diag(DD1)
    print('DD1:')
    print(DD1)

    # Nova fundamentalna matrica, FF1, dobijena koriscenjem DD1.
    FF1 = U @ DD1 @ VT
    print('FF1:')
    print(FF1)

    # Nadalje cemo koristiti FF1, jer su U i V iste za FF i FF1, pa
    # su im tako isti i epipolovi, ali FF1 ima determinantu blizu nuli,
    # sto je bolje.
    det1 = LA.det(FF1)
    print('det1: ', det1)

    ##########################################
    ##### Rekonstukcija skrivenih tacaka #####
    ##########################################

    # Nevidljive tacke prve projekcije
    x5 = findInvisible(x4, x8, x6, x2, x1,
                x1, x4, x3, x2, x8)

    x13 = findInvisible( x9, x10, x11, x12, x14,
                x11, x15, x10, x14,  x9)

    x16 = findInvisible(x10, x14, x11, x15, x12,
                x9, x10, x11, x12, x15)

    # Nevidljive tacke druge projekcije
    y5 = findInvisible(y4, y8, y6, y2, y1,
                y1, y4, y3, y2, y8)

    y13 = findInvisible(y9, y10, y11, y12, y14,
                y11, y15, y10, y14, y9)

    y16 = findInvisible(y10, y14, y11, y15, y12,
                y9, y10, y11, y12, y15)

    #########################
    ##### Triangulacija #####
    #########################

    # Kanonska matrica kamere
    T1 = np.hstack([np.eye(3), np.zeros(3).reshape(3, 1)])

    # Matrica epipola e2.
    # Funkcija vec vraca matricu vektorskog mnozenja kao kod Rodrigeza.
    E2 = vec(e2)

    # Druga matrica kamere
    T2 = np.hstack([E2 @ FF1, e2.reshape(3, 1)])

    # Za svaku tacku dobijamo sistem od 4 jednacije sa 4 homogene nepoznate
    jednacine = lambda xx, yy: np.array([ xx[1]*T1[2] - xx[2]*T1[1],
                                          -xx[0]*T1[2] + xx[2]*T1[0],
                                          yy[1]*T2[2] - yy[2]*T2[1],
                                          -yy[0]*T2[2] + yy[2]*T2[0]])

    # Afine 3D koordinate
    UAfine = lambda xx: (xx/xx[-1])[:-1]

    # Funkcija koja vraca 3D koordinate rekonstruisane tacke,
    # koristeci prethodni princip
    TriD = lambda xx, yy: UAfine(LA.svd(jednacine(xx, yy))[-1][-1])

    # Piksel koordinate sa obe slike
    slika1 = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9,
                        x10, x11, x12, x13, x14, x15, x16])
    
    slika2 = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9,
                        y10, y11, y12, y13, y14, y15, y16])

    # Rekonstruisane 3D koordinate tacaka
    rekonstruisane = np.array([TriD(x, y) for x, y
                                in zip(slika1, slika2)])

    # Mnozenje z-koordinate, posto nije bilo normalizacije
    rekonstruisane400 = np.array([*map(lambda x:
                                        np.diag([1, 1, 400]) @ x,
                                        rekonstruisane)])

    # Vracanje rezultata (izbaceno je konstruisanje ivica, detaljnije
    # objasnjenje zasto je tako je u funkciji plot).
    return rekonstruisane400

def plot(rek):
    # Inicijalizujemo grafik.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Icrtavamo temena na grafiku.
    # rek[:, 0], rek[:, 1], rek[:, 2] nam redom vracaju nizove svih
    # x, y i z koordinata. To je ulaz za scatter3D.
    ax.scatter3D(rek[:, 0], rek[:, 1], rek[:, 2])

    # U uputstvu za zadatak funkcija vraca i ivice. Medjutim, nama je za 
    # ovu bilioteku potrebno da zadamo strane, tako da je to jedna izmena
    # u odnosu na uputstvo. Temena koja koristimo ovde su zapravo temena 
    # sa slike, samo sa indeksom 'indeks-1'.
    verts1 = [[rek[0], rek[1], rek[2], rek[3]],
              [rek[1],rek[5],rek[6],rek[2]], 
              [rek[3], rek[2], rek[6], rek[7]], 
              [rek[0],rek[1],rek[5],rek[4]], 
              [rek[0],rek[4],rek[7],rek[3]],
              [rek[4],rek[5],rek[6],rek[7]]]

    verts2 = [[rek[8], rek[9], rek[10], rek[11]],
              [rek[8],rek[9],rek[13],rek[12]], 
              [rek[9], rek[10], rek[14], rek[13]], 
              [rek[8],rek[11],rek[15],rek[12]], 
              [rek[12],rek[13],rek[14],rek[15]],
              [rek[10],rek[11],rek[15],rek[14]]]

    # Iscrtavamo tela na grafiku.
    ax.add_collection3d(Poly3DCollection(verts1, 
    facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.add_collection3d(Poly3DCollection(verts2, 
    facecolors='purple', linewidths=1, edgecolors='r', alpha=.25))

    # Postavljamo imena osa.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Prikazujemo grafik.
    plt.show()

if __name__ == "__main__":
    rek = reconstruct()

    print('-----------------------------------------------')
    print('Koordinate rekonstruisanih tacaka:')
    print(rek)

    plot(rek)