import numpy as np

def afinize(x):
    return list(map(lambda a : a/x[2], x))

def nevidljivo(p1, p2, p3, p5, p6, p7, p8):
    Xb = np.cross(np.cross(p2 + [1], p6 + [1]), np.cross(p1 + [1], p5 + [1]))
    Yb = np.cross(np.cross(p5 + [1], p6 + [1]), np.cross(p7 + [1], p8 + [1]))

    p4 = np.cross(np.cross(p8 + [1], Xb), np.cross(p3 + [1], Yb))

    return list(map(int, map(round, afinize(p4))))[:2]

def main():
    # Kooridate temena na mojoj slici.
    p1 = [182, 271]
    p2 = [715, 70]
    p3 = [800, 223]
    p5 = [93, 474]
    p6 = [714, 330]
    p7 = [818, 471]
    p8 = [250, 550]


    # Koordinate slike iz materijala koje ja uspevam da ocitam u Inkscape-u.
    # Nisam uspela da ih okrenem, razlikuje se po tome sto je u Paintu (0,0)
    # gore levo, a ovde je dole levo.
    # p1 = [595, 270, 1]
    # p2 = [292, 50, 1]
    # p3 = [157, 188, 1]
    # p5 = [665, 454, 1]
    # p6 = [304, 275, 1]
    # p7 = [135, 398, 1]
    # p8 = [509, 521, 1]

    # Zakomentarisati tacke iznad, a otkomentarisati ove ispod za
    # isprobavanje test primera datog u materijalima. Dobija se tacan
    # rezultat, tj. (595, 301)
    # p1 = [595, 301, 1]
    # p2 = [292, 517, 1]
    # p3 = [157, 379, 1]
    # p5 = [665, 116, 1]
    # p6 = [304, 295, 1]
    # p7 = [135, 163, 1]
    # p8 = [509, 43, 1]

    print(nevidljivo(p1,p2,p3,p5,p6,p7,p8))

if __name__ == '__main__':
    main()
