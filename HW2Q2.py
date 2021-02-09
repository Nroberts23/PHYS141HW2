"""
Name: Nathan Roberts
PID: A14384608
"""
#imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from celluloid import Camera
from mpl_toolkits.mplot3d import Axes3D

MAXPNT = 100

def main(print_imgs=False):
	#initial conditions

	#x = np.array([np.zeros(n)]) #set initial position as 0 as default for all points
	#v = np.array([np.zeros(n)]) #set initial velocity as 0 as default for all points
	v_orb = 0.01721420632

	tnow = 0 #set initial time
	t_history = []
	
	sun_x = np.array([float(0), float(0), float(0)])
	sun_v = np.array([float(0), float(0), float(0)])

	one_x = np.array([float(1), float(0), float(0)])
	one_v = np.array([float(0), float(v_orb), float(0)])

	two_x = np.array([float(0), float(1), float(0)])
	two_v = np.array([float(-v_orb), float(0), float(0)])

	three_x = np.array([float(-1), float(0), float(0)])
	three_v = np.array([float(0), float(-v_orb), float(0)])

	four_x = np.array([float(0), float(-1), float(0)])
	four_v = np.array([float(v_orb), float(0), float(0)])

	x = np.vstack((sun_x, one_x, two_x, three_x, four_x))
	v = np.vstack((sun_v, one_v, two_v, three_v, four_v))

	orb_mass = 3
	masses = np.array([1, orb_mass, orb_mass, orb_mass, orb_mass])
	n = len(x)
	
	x_history = [[i] for i in x]
	v_history = [[i] for i in v]
	t_history.append(tnow)
	#integration perameters
	max_step = 2000
	nout = 10
	dt = 2

	#looping to perform integration
	for i in range(max_step):
		if (i % nout == 0): #if enough steps have passed, print the state
			if(print_imgs):
				printstate(x, x_history, n, tnow)
			x_history = np.append([[i] for i in x], x_history, axis=1)
			v_history = np.append([[i] for i in v], v_history, axis=1)
			t_history.append(tnow)


		x, v = leapstep(x, v, n, dt, masses) #take an integration step
		
		tnow += dt

	if (max_step % nout == 0): #if the last step would have printed
		if(print_imgs):
			printstate(x, x_history, n, tnow) #then print
		x_history = np.append([[i] for i in x], x_history, axis=1)
		v_history = np.append([[i] for i in v], v_history, axis=1)
		t_history.append(tnow)


	return x_history, v_history, t_history

def leapstep(x, v, n, dt, masses):
	a = acc(x, n, masses) #call the acceleration code

	for i in range(n):
		v[i] = v[i] + 0.5 * dt * a[i] #loop over all points and increase the velocities by a half setp

	for i in range(n): #loop again an increase the positions by a full step
		x[i] = x[i] + dt * v[i]

	a = acc(x, n, masses) #call the acceleration code again
	
	for i in range(n):
		v[i] = v[i] + 0.5 * dt * a[i] #another loop through velocity half-stepping

	return x, v

def acc(x, n, masses):
	GM =  0.0002959 #G * M in AU^3 / day^2

	a = []
	for p in range(n):
		pos = x[p]
		others = [x[i] for i in range(n) if not (i == p)]
		o_mass = [masses[i] for i in range(n) if not (i==p)]

		a_comps = []
		for j in range(len(others)):
			dist_inv = 1 / np.linalg.norm(pos-others[j])
			k = GM * o_mass[j]	* dist_inv * dist_inv * dist_inv

			a_comps.append(k * (others[j] - pos) / dist_inv)

		a.append(sum(a_comps))

	return a




nonlin_pen = lambda x: [-np.sin(i) for i in x]

def printstate(x, x_h, n, tnow):
	#point_history.append(x[0])
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(-1.2,1.2)
	ax.set_ylim(-1.2,1.2)

	ax.set_title ("Ring Orbits: Unstable")
	for i in range(n):
		ax.plot([x[i][0]], [x[i][1]], [x[i][2]], 'ob')

	plt.savefig('animate/unstable_ring_' + str(tnow) + '.png')
"""
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(-1.2,1.2)
	ax.set_ylim(-1.2,1.2)
	ax.plot([0],[0],[0], 'oy')

	for planet in x_history:
		xs = [i[0] for i in planet]
		ys = [i[1] for i in planet]
		zs = [i[2] for i in planet]
		
		ax.plot(xs=xs, ys=ys, zs=zs)
		#ax.plot(xs=x_history[i][-1], ys=v_history[i][-1], zs=t_history[-1])

	ax.set_title('Orbits of the Planets around the Sun (Origin)')"""
		

def plot_2d(x_history):
	fig = plt.figure()
	plt.plot(0,0, 'oy')

	for planet in x_history:
		xs = [i[0] for i in planet]
		ys = [i[1] for i in planet]
		plt.plot(xs, ys)

	plt.title('Orbits of the Planets Around the Sun')

def plot_3d(x_history, v_history, t_history):
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(-1.5,1.5)
	ax.set_ylim(-1.5, 1.5)
	ax.plot([0],[0],[0], 'oy')

	for planet in x_history:
		xs = [i[0] for i in planet]
		ys = [i[1] for i in planet]
		zs = [i[2] for i in planet]
		
		ax.plot(xs=xs, ys=ys, zs=zs)
		#ax.plot(xs=x_history[i][-1], ys=v_history[i][-1], zs=t_history[-1])

	ax.set_title('Orbits of the Planets around the Sun (Origin)')
