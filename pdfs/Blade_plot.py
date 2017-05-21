#!/usr/bin/python

import math
import numpy as np
import pylab as pl

SS = np.genfromtxt('BladeSS.txt')
PS = np.genfromtxt('BladePS.txt')

case = 'TBLADE_CONST'
var ='V'

V_g1_12 = np.genfromtxt('%s/X0.005/%s12.txt'%(case,var))
V_g2_12 = np.genfromtxt('%s/X0.02/%s12.txt'%(case,var))
V_g3_12 = np.genfromtxt('%s/X0.04/%s12.txt'%(case,var))
V_g4_12 = np.genfromtxt('%s/X0.05/%s12.txt'%(case,var))
V_g5_12 = np.genfromtxt('%s/X0.06/%s12.txt'%(case,var))
V_g6_12 = np.genfromtxt('%s/X0.07/%s12.txt'%(case,var))

def curv_abscissa(A):

	# function to compute curvilinear abscissa of a curve

	n = np.shape(A)
	N = n[0]
	s = [0]*N

	for k in range(N-1):

		s[k+1] = s[k] + math.sqrt((A[k,0] - A[k+1,0])**2 + (A[k,1] - A[k+1,1])**2)


	return s


s_SS = curv_abscissa(SS)


def find_point(B,s):

	# function to compute the (x,y) coordinates corresponding to a given curvilinear abscissa
	# B curvilinear abscissa
	# x target point in terms of curvilinear abscissa

	n = np.shape(B)
	N = n[0]
	index = 0

	for i in range(N):

			if s-0.0005 <= B[i] <= s + 0.0005:

				index = i


	return index

#points

x_1 = 0.005
x_2 = 0.02
x_3 = 0.04
x_4 = 0.05
x_5 = 0.06
x_6 = 0.07

index_xy1 = find_point(s_SS,x_1)
index_xy2 = find_point(s_SS,x_2)
index_xy3 = find_point(s_SS,x_3)
index_xy4 = find_point(s_SS,x_4)
index_xy5 = find_point(s_SS,x_5)
index_xy6 = find_point(s_SS,x_6)	

# define array of points

SS_points = np.array([SS[index_xy1,:],SS[index_xy2,:],SS[index_xy3,:],SS[index_xy4,:],SS[index_xy5,:],SS[index_xy6,:]])				

def normal(A,i):

	# angular coeff tang

	t = (A[i+1,1] - A[i-1,1])/(A[i+1,0] - A[i-1,0])

	#angular coeff normal 

	n = -1/t

	return n


n_1 = normal(SS,index_xy1)
n_2 = normal(SS,index_xy2)
n_3 = normal(SS,index_xy3)
n_4 = normal(SS,index_xy4)
n_5 = normal(SS,index_xy5)
n_6 = normal(SS,index_xy6)


x_1 = np.arange(SS[index_xy1,0]-0.015,SS[index_xy1,0]+0.002,0.001)
x_2 = np.arange(SS[index_xy2,0],SS[index_xy2,0]+0.015,0.001)
x_3 = np.arange(SS[index_xy3,0],SS[index_xy3,0]+0.015,0.001)
x_4 = np.arange(SS[index_xy4,0],SS[index_xy4,0]+0.015,0.001)
x_5 = np.arange(SS[index_xy5,0],SS[index_xy5,0]+0.015,0.001)
x_6 = np.arange(SS[index_xy6,0],SS[index_xy6,0]+0.015,0.001)


y_1 = SS[index_xy1,1] + n_1*(x_1-SS[index_xy1,0])
y_2 = SS[index_xy2,1] + n_2*(x_2-SS[index_xy2,0])
y_3 = SS[index_xy3,1] + n_3*(x_3-SS[index_xy3,0])
y_4 = SS[index_xy4,1] + n_4*(x_4-SS[index_xy4,0])
y_5 = SS[index_xy5,1] + n_5*(x_5-SS[index_xy5,0])
y_6 = SS[index_xy6,1] + n_6*(x_6-SS[index_xy6,0])


############## Rotation of The Profiles ################

def rot(R,n,S,index):
	
	m = np.shape(R)
	N = m[0]
	
	
	######### rotation matrix ##########
	if n < 0:
		
		alpha = math.atan(n) 

	else:

		alpha = math.atan(-n) - math.pi*(0.5+0.15)

	M11 = math.cos(alpha)
	M12 = math.sin(alpha)

	############# Rotation ################
	
	M = np.matrix([[M11, -M12],[M12, M11]])

	V_t  = np.transpose(R)

	x_old = V_t[0,:]/(V_t[0,N-1]*10)
	y_old = V_t[1,:]/(V_t[1,N-1]*200)

	R_old = np.vstack((x_old,y_old))

	R_t = M*R_old
	
	R_new = np.transpose(R_t)

############# rigid translation ##########
	
	x_shift = np.ones(N)*S[index,0]
	y_shift = np.ones(N)*S[index,1]

	M_s = np.vstack((x_shift,y_shift))

	R_new = R_new + np.transpose(M_s)

###################################

	return R_new

V_rot_1 = rot(V_g1_12,n_1,SS,index_xy1)
V_rot_2 = rot(V_g2_12,n_2,SS,index_xy2)
V_rot_3 = rot(V_g3_12,n_3,SS,index_xy3)
V_rot_4 = rot(V_g4_12,n_4,SS,index_xy4)
V_rot_5 = rot(V_g5_12,n_5,SS,index_xy5)
V_rot_6 = rot(V_g6_12,n_6,SS,index_xy6)

p = 50  # max index n axis

f1 = pl.figure(1)
pl.rcParams['axes.facecolor'] = 'white'
f1.patch.set_facecolor('white')

pl.plot(SS[:,0],SS[:,1],'black',linewidth=5.0) #SS plot
pl.plot(PS[:,0],PS[:,1],'black',linewidth=5.0) #PS plot

#pl.plot(SS_points[2:6,0],SS_points[2:6,1] ,'ro',markersize=15) #control point plot

pl.plot(x_1,y_1,'k--',linewidth=1)	#axis plot
pl.plot(x_2,y_2,'k--',linewidth=1)
pl.plot(x_3,y_3,'k--',linewidth=1)
pl.plot(x_4,y_4,'k--',linewidth=1)
pl.plot(x_5,y_5,'k--',linewidth=1)
pl.plot(x_6,y_6,'k--',linewidth=1)


#pl.plot(V_rot_1[0:p,0],V_rot_1[0:p,1],'b',linewidth=3.0)
#pl.plot(V_rot_2[0:p,0],V_rot_2[0:p,1],'b',linewidth=3.0)
#pl.plot(V_rot_3[0:p,0],V_rot_3[0:p,1],'b',linewidth=3.0)
#pl.plot(V_rot_4[0:p,0],V_rot_4[0:p,1],'b',linewidth=3.0)
#pl.plot(V_rot_5[0:p,0],V_rot_5[0:p,1],'b',linewidth=3.0)
#pl.plot(V_rot_6[0:p,0],V_rot_6[0:p,1],'b',linewidth=3.0)

pl.xlim([-0.05,0.09])
pl.ylim([-0.02,0.06])

pl.xticks([ ])
pl.yticks([ ])

ax = pl.gca()
ax.invert_yaxis()
pl.show()

#output coord
'''
print('point number 3 wall',x_3[0],y_3[0])
print('point number 3 inf',x_3[np.shape(x_3)[0]-1],y_3[np.shape(y_3)[0]-1])

print('point number 4 wall',x_4[0],y_4[0])
print('point number 4 inf',x_4[np.shape(x_4)[0]-1],y_4[np.shape(y_4)[0]-1])

print('point number 5 wall',x_5[0],y_5[0])
print('point number 5 inf',x_5[np.shape(x_5)[0]-1],y_5[np.shape(y_5)[0]-1])

print('point number 6 wall',x_6[0],y_6[0])
print('point number 6 inf',x_6[np.shape(x_6)[0]-1],y_6[np.shape(y_6)[0]-1])
'''
#writing coord

size = np.shape(x_1)[0]-2

xlist_0 = [x_1[0],x_2[0],x_3[0], x_4[0] ,x_5[0] ,x_6[0]]
ylist_0 = [y_1[0],y_2[0],y_3[0], y_4[0] ,y_5[0] ,y_6[0]]

xlist_fin = [x_1[size-1],x_2[size-1],x_3[size-1], x_4[size-1] ,x_5[size-1] ,x_6[size-1]]
ylist_fin = [y_1[size-1],y_2[size-1],y_3[size-1], y_4[size-1] ,y_5[size-1] ,y_6[size-1]]

print('initial point x_coord,y_coord')
print(np.transpose(np.matrix([xlist_0,ylist_0])))
print('final point x_coord,y_coord')
print(np.transpose(np.matrix([xlist_fin,ylist_fin])))
Coord = np.transpose(np.matrix([xlist_fin,ylist_fin]))

f2 = pl.figure(2)
pl.rcParams['axes.facecolor'] = 'white'
f2.patch.set_facecolor('white')

pl.plot(SS[:,0],SS[:,1],'black',linewidth=5.0) #SS plot
pl.plot(PS[:,0],PS[:,1],'black',linewidth=5.0) #PS plot

pl.plot(x_1[size-1],y_1[size-1],'ro',x_1[0],y_1[0],'bo')
pl.plot(x_2[0],y_2[0],'ro',x_2[size-1],y_2[size-1],'bo')
pl.plot(x_3[0],y_3[0],'ro',x_3[size-1],y_3[size-1],'bo')
pl.plot(x_4[0],y_4[0],'ro',x_4[size-1],y_4[size-1],'bo')
pl.plot(x_5[0],y_5[0],'ro',x_5[size-1],y_5[size-1],'bo')

ax = pl.gca()
ax.invert_yaxis()

pl.axis('equal')

pl.show()
#np.savetxt('test.txt', Coord)
