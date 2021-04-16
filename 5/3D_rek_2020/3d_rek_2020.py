###############################
##### Rekonstrukcija 2020 #####
###############################

import numpy as np
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

# Piksel koordinate svih tacaka (od toga cemo birati 8 za vektore xx i yy).
x1 = np.array([819, 111, 1.])
x2 = np.array([957, 162, 1.])
x3 = np.array([995, 125, 1.])
x4 = np.array([859, 80, 1.])
x5 = np.array([796, 304, 1.])
x6 = np.array([920, 358, 1.])
x7 = np.array([956, 320, 1.])
# x8-?

x9 = np.array([327, 345, 1.])
x10 = np.array([458, 370, 1.])
x11 = np.array([515, 273, 1.])
x12 = np.array([391, 251, 1.])
x13 = np.array([370, 559, 1.])
x14 = np.array([484, 584, 1.])
x15 = np.array([530, 484, 1.])
# x16 - ?

x17 = np.array([144, 550, 1.])
x18 = np.array([440, 750, 1.])
x19 = np.array([820, 380, 1.])
x20 = np.array([550, 253, 1.])
x21 = np.array([182, 655, 1.])
x22 = np.array([450, 860, 1.])
x23 = np.array([810, 490, 1.])
# x24 -?

y1 = np.array([920, 446, 1.])
y2 = np.array([818, 560, 1.])
y3 = np.array([923, 612, 1.])
y4 = np.array([1016, 491, 1.])
# y5 - ?
y6 = np.array([780, 771, 1.])
y7 = np.array([867, 822, 1.])
y8 = np.array([960, 703, 1.])

y9 = np.array([320, 74, 1.])
y10 = np.array([257, 121, 1.])
y11 = np.array([375, 137, 1.])
y12 = np.array([417, 90, 1.])
# y13 - ?
y14 = np.array([295, 327, 1.])
y15 = np.array([401, 342, 1.])
y16 = np.array([439, 286, 1.])

# y17 - ?
y18 = np.array([143, 322, 1.])
y19 = np.array([531, 525, 1.])
y20 = np.array([747, 349, 1.])
# y21 - ?
y22 = np.array([167, 426, 1.])
y23 = np.array([538, 643, 1.])
y24 = np.array([737, 450, 1.])

# Vektori tih tacaka, koje cemo proslediti funkciji (bez normalizacije)
# Najbolji izlaz
# xx = np.array([x1, x2, x3, x10, x11, x18, x19, x20])
# yy = np.array([y1, y2, y3, y10, y11, y18, y19, y20])

# Primeri losijeg izlaza
# xx = np.array([x1, x2, x3, x4, x18, x19, x20, x22])
# yy = np.array([y1, y2, y3, y4, y18, y19, y20, y22])

xx = np.array([x1, x2, x3, x4, x9, x10, x11, x12])
yy = np.array([y1, y2, y3, y4, y9, y10, y11, y12])


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
    x8 = findInvisible(x1, x5, x7, x3, x4,
                x4, x1, x7, x6, x5)

    x16 = findInvisible(x9, x10, x13, x14, x15,
                x9, x13, x11, x15, x12)

    x24 = findInvisible(x21, x22, x20, x19, x23,
                x17, x20, x22, x23, x21)

    # Nevidljive tacke druge projekcije
    y5 = findInvisible(y4, y3, y8, y7, y6,
                y2, y6, y4, y8, y1)

    y13 = findInvisible(y10, y14, y12, y16, y9,
                y12, y9, y14, y15, y16)
    
    y17 = findInvisible(y18, y19, y22, y23, y20,
                y19, y20, y24, y23, y18)

    y21 = findInvisible(y22, y23, y17, y20, y24,
                y18, y17, y24, y23, y22)

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
    slika1 = np.array([x1, x2, x3, x4, x5, x6, x7, x8, 
                        x9, x10, x11, x12, x13, x14, x15, x16,
                        x17, x18, x19, x20, x21, x22, x23, x24])
    
    slika2 = np.array([y1, y2, y3, y4, y5, y6, y7, y8, 
                        y9, y10, y11, y12, y13, y14, y15, y16,
                        y17, y18, y19, y20, y21, y22, y23, y24])

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

    verts3 = [[rek[16], rek[17], rek[18], rek[19]],
              [rek[16],rek[17],rek[21],rek[20]], 
              [rek[17], rek[18], rek[22], rek[21]], 
              [rek[20],rek[21],rek[22],rek[23]], 
              [rek[16],rek[19],rek[23],rek[20]],
              [rek[18],rek[19],rek[23],rek[22]]]

    # Iscrtavamo tela na grafiku.
    ax.add_collection3d(Poly3DCollection(verts1, 
    facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.add_collection3d(Poly3DCollection(verts2, 
    facecolors='purple', linewidths=1, edgecolors='r', alpha=.25))

    ax.add_collection3d(Poly3DCollection(verts3, 
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
    print('Afine koordinate rekonstruisanih tacaka:')
    print(rek)

    plot(rek)