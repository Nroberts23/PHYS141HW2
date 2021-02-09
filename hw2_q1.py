"""
Name: Nathan Roberts
PID: A14384608
"""
#imports
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from celluloid import Camera
from mpl_toolkits.mplot3d import Axes3D

MAXPNT = 100

def main(print_imgs=False):
	#initial conditions

	#x = np.array([np.zeros(n)]) #set initial position as 0 as default for all points
	#v = np.array([np.zeros(n)]) #set initial velocity as 0 as default for all points
	tnow = 0 #set initial time
	t_history = []
	
	#earth
	x = np.array([0, -1])

	#0.019182303

	v = np.array([float(0.018899), float(0)])
	n = len(x)
	
	x_history = x.reshape(1,2)
	v_history = v.reshape(1,2)
	t_history.append(tnow)
	#integration perameters
	max_step = 259	
	nout = 4
	dt = 1

	#looping to perform integration
	for i in range(max_step):
		if (i % nout == 0): #if enough steps have passed, print the state
			x_history = np.append([x], x_history, axis=0)
			v_history = np.append([v], v_history, axis=0)
			t_history.append(tnow)
			if(print_imgs):
				printstate(x, x_history, n, tnow)

		x, v = leapstep(x, v, n, dt) #take an integration step
		
		tnow += dt

	if (max_step % nout == 0): #if the last step would have printed
		 #then print
		x_history = np.append([x], x_history, axis=0)
		v_history = np.append([v], v_history, axis=0)
		t_history.append(tnow)
		if(print_imgs):
			printstate(x, x_history, n, tnow)


	return x_history, v_history, t_history

def leapstep(x, v, n, dt):
	a = acc(x, n) #call the acceleration code

	v = v + 0.5 * dt * a #loop over all points and increase the velocities by a half step
	x = x + dt * v

	a = acc(x, n) #call the acceleration code again
	
	for i in range(n):
		v[i] = v[i] + 0.5 * dt * a[i] #another loop through velocity half-stepping

	return x, v

def acc(x, n):
	GM = 0.0002959 #G * M in AU^3 / day^2
	rinv = (sum(x ** 2)) ** (-0.5)
	k = GM * rinv * rinv * rinv #GM / r^2

	return -k * x
	#return -k * (x **2 / (rinv * rinv))



nonlin_pen = lambda x: [-np.sin(i) for i in x]

def printstate(x, x_h, n, tnow):
	#point_history.append(x[0])
	fig = plt.figure(figsize=(7, 7))
	earth_orb = plt.Circle((0,0), 1, fill=False)
	mars_orb = plt.Circle((0,0), 1.524, fill=False)

	ax = fig.add_subplot(111)
	ax.set_title ("Transfer Orbit from Earth to Mars")
	ax.set_xlim(-1.75, 1.75)
	ax.set_ylim(-1.75, 1.75)

	plt.figtext(.5,.6,'Time = ' + str(tnow) + ' (days)', fontsize=12, ha='center')
	ax.plot([0],[0], 'oy')
	
	xs = [j[0] for j in x_h]
	ys = [j[1] for j in x_h]
	ax.plot(xs, ys, color = 'blue')
	ax.plot([x[0]], [x[1]], 'ob')
	ax.add_artist(earth_orb)
	ax.add_artist(mars_orb)

	plt.savefig('animate/orbit_' + str(tnow) + '.png')

		

def plot_2d(x_history):
	
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(111)
	ax.plot(0,0, 'oy')
	
	xs = [i[0] for i in x_history]
	ys = [i[1] for i in x_history]

	earth_orb = plt.Circle((0,0), 1, fill=False)
	mars_orb = plt.Circle((0,0), 1.524, fill=False)

	ax.plot(xs, ys)
	ax.add_artist(earth_orb)
	ax.add_artist(mars_orb)
	ax.set_xlim(-2,2)
	ax.set_ylim(-2,2)

	plt.title('Transfer Orbit From Earth to Mars')

def plot_3d(x_history, v_history, t_history):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot([0],[0],[0], 'oy')
	for planet in x_history:
		xs = [i[0] for i in planet]
		ys = [i[1] for i in planet]
		zs = [i[2] for i in planet]
		
		ax.plot(xs=xs, ys=ys, zs=zs)

		#ax.plot(xs=x_history[i][-1], ys=v_history[i][-1], zs=t_history[-1])

	ax.set_title('Orbits of the Planets around the Sun (Origin)')

def do_plots():
	x, v, t = main()
	plot_2d(x)

