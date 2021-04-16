# Vecina koda za OpenGL inicijalizaciju, pocetni parametri svetla, tajmera, 
# pocetno iscrtavanje koordinata, callback funkcije (onReshape, onKeyboard...)
# i slicno, preuzeta je iz materijala za predmet Racunarska Grafika, od profesora Ivana Cukica (tada asistenta).

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

import math
import funkcije as izometrije


class Globals:
    def __init__(self):
        self.phi1 = -3*math.pi/4
        self.theta1 = math.pi/4
        self.psi1 = 2*math.pi/5
        self.phi2 = 2*math.pi/3
        self.theta2 = -3*math.pi/7
        self.psi2 = math.pi/4

        # Koordinate pocetne kocke
        self.x1 = 6
        self.y1 = 0
        self.z1 = 0

        # Koordinate krajnje kocke
        self.x2 = -4
        self.y2 = 2
        self.z2 = -3

        # Parametri za SLERP
        self.t = 0
        self.tm = 100
        self.q1 = []
        self.q2 = []

        # Tajmer koji oznacava da li je animacija u toku
        self.timer_active = False

g = Globals()

# Funkcija koja iscrtava svetske koordinatne ose
def drawCoordinates(size):
    glBegin(GL_LINES)
    glColor3f (1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(size, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, size, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, size)
    glEnd()

def onDisplay():
    global g

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
  
    # Inicijalizacija kamere i pogleda
    gluLookAt(10, 15, 15,
              0,  0,  0,
              0,  1,  0)

    # Svetske koordinatne ose
    drawCoordinates(50)

    # Iscrtavanje pocetne kocke
    glColor3f(1.0, 0.0, 0.0)
    drawCube(g.x1, g.y1, g.z1, g.phi1, g.theta1, g.psi1)
    
    # Iscravanje krajnje kocke
    glColor3f(0.0, 0.0, 1.0)
    drawCube(g.x2, g.y2, g.z2, g.phi2, g.theta2, g.psi2)
    
    # Pokrecemo animaciju
    slerpAnimation()

    glutSwapBuffers()


def drawCube(x, y, z, phi, theta, psi):
    glPushMatrix()

    glTranslatef(x, y, z)

    # Izracunavamo sopstvene ose kocke.
    A = izometrije.euler2a(phi, theta, psi)
    p, alpha = izometrije.axis_angle(A)

    glRotatef(alpha / math.pi * 180, p[0], p[1], p[2])

    glutSolidCube(2)
    drawCoordinates(3)

    glPopMatrix()


def slerpAnimation():
    global g
  
    glPushMatrix()
    glColor3f(1.0, 1.0, 0.0)

    x = (1 - g.t / g.tm) * g.x1 + (g.t / g.tm) * g.x2
    y = (1 - g.t / g.tm) * g.y1 + (g.t / g.tm) * g.y2
    z = (1 - g.t / g.tm) * g.z1 + (g.t / g.tm) * g.z2

    glTranslatef(x,y,z)
    q = izometrije.slerp(g.q1, g.q2, g.tm, g.t)
    p, phi = izometrije.q2axisangle(q)

    glRotatef(phi / math.pi * 180, p[0], p[1], p[2])

    glutWireCube(2)
    drawCoordinates(4)

    glPopMatrix()


def onReshape(w, h):
    glViewport(0, 0, w, h)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, float(w) / h, 1, 1500)


def onKeyboard(ch, x, y):
	global g

	if ord(ch) == 27:
		sys.exit(0)
	elif ord(ch) == ord('g') or ord(ch) == ord('G'):
		if not g.timer_active:
			glutTimerFunc(40, onTimer, 0)
			g.timer_active = True

  

def onTimer(value):
    global g

    g.t += 1

    # Ukoliko smo stigli do kraja, treba zaustaviti animaciju
    if g.t > g.tm:
        g.timer_active = False
        return

    glutPostRedisplay()

    if g.timer_active:
        glutTimerFunc(40, onTimer, 0)


def main():
    global g

    glutInit()
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("SLERP")
    glClearColor(0.1,0.1,0.1,0)
    glutDisplayFunc(onDisplay)
    glutReshapeFunc(onReshape)
    glutKeyboardFunc(onKeyboard)

    glEnable(GL_DEPTH_TEST)
    glLineWidth(3)
    
    # inicijalizacija kvaterniona
    A = izometrije.euler2a(g.phi1, g.theta1, g.psi1)
    p, alpha = izometrije.axis_angle(A)
    g.q1 = izometrije.axisangle2q(p, alpha)

    A = izometrije.euler2a(g.phi2, g.theta2, g.psi2)
    p, alpha = izometrije.axis_angle(A)
    g.q2 = izometrije.axisangle2q(p, alpha)

    glutMainLoop()


if __name__ == '__main__':
	main()
