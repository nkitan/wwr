import numpy as np
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing(use_latex='mathjax',pretty_print=False)
from sympy import  *

# function to calculate force magnitudes
def susp_forces(coordinates, force_vector):
    F = force_vector
    O = coordinates[0]
    A = coordinates[1]
    B = coordinates[2]
    Q = coordinates[3]
    D = coordinates[4]
    E = coordinates[5]
    K = coordinates[6]
    G = coordinates[7]
    H = coordinates[8]
    I = coordinates[9]
    J = coordinates[10]
    
    # Unit vectors
    u_1 = Matrix(G - A).normalized()
    u_2 = Matrix(K - A).normalized()
    u_3 = Matrix(E - B).normalized()
    u_4 = Matrix(D - B).normalized()
    u_5 = Matrix(J - I).normalized()
    u_6 = Matrix(H - Q).normalized()
    unit_vectors = [u_1, u_2, u_3, u_4, u_5, u_6]

    # r vectors
    r_OG = Matrix(G - O)
    r_OE = Matrix(E - O)
    r_OD = Matrix(D - O)
    r_OK = Matrix(K - O)
    r_OH = Matrix(H - O)
    r_OJ = Matrix(J - O)
    r_vectors = [r_OG, r_OK, r_OE, r_OD, r_OJ, r_OH]

    # Building A and B matrix

    A_upper = zeros(3,6)
    for j in range(6):
        for i in range(3):
            A_upper[i,j] = float(unit_vectors[j][i])

    A_lower = zeros(3,6)
    for j in range(6):
        for i in range(3):
            A_lower[i,j] = float(r_vectors[j].cross(unit_vectors[j])[i])/1000

    A = Matrix([[A_upper],[A_lower]])     
    B = Matrix([F[0], F[1], F[2], 0, 0, 0])

    # Solve x for A and B
    x = A.inv()*B

    # Printing reaction forces in [N]
    
    return x, Matrix(np.array(symbols('R_G R_K R_E R_D R_J R_H')))

# function to plot diagrams
def plot_vector(point_1, point_2, color_vector, label):
    x_vector = np.array([point_1[0], point_2[0]])
    y_vector = np.array([point_1[1], point_2[1]])
    z_vector = np.array([point_1[2], point_2[2]])
    
    ax0.plot(x_vector, -y_vector, label=label)
    ax1.plot(x_vector, z_vector,  label=label)
    ax2.plot(y_vector,z_vector,  label=label)
    
    ax0.scatter(x_vector, -y_vector, color='k')
    ax1.scatter(x_vector, z_vector, color='k')
    ax2.scatter(y_vector,z_vector, color='k')
    
    ax0.set_title('upper view')
    ax1.set_title('lateral view')
    ax2.set_title('front view')
    
    ax0.set_aspect('equal')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    ax2.legend()

# util function to get an array
def get_array():
    list = []

    for i in range(0,3):
        temp = int(input())
        list.append(temp)
    
    return np.array(list)


# EXECUTION STARTS HERE

# force vector [N]
force_vector = np.array([0, 0, 2000]) # force on tire along x, y, z axes

'''
print("ENTER FORCE ARRAY: ") 
force_vector = get_array()
'''
# Suspension hardpoint x,y,z coordinates [mm]
O = np.array([688.01, 581.77, 19.297])    # origin

Q = np.array([618.636, 515.149, 177.997]) # Tie rod
H = np.array([650.677, 105.501, 199.891]) # Tie Rod

B = np.array([705.539, 468.673, 317.419]) # upper wishbone ball joint
E = np.array([566.520, 254.836, 303.502]) # Upper Wishbone Contact towards front 
D = np.array([821.874, 317.636, 333.584]) # Upper wishbone Contact towards rear


A = np.array([686.254, 494.591, 130.327]) # lower wishbone ball joint
G = np.array([570.364, 261.602, 126.679]) # Lower wishbone Contact towards front
K = np.array([878.68, 266.265, 126.902])  # Lower wishbone Contact towards rear

I = np.array([708.706,439.173,276.889])   # Pushrod lower 
J = np.array([689.133,423.238,141.498])   # Pushrod Upper

# Angles
AG = ([129.56,39.57,90.74])
AK = ([27.00,63.02,90.95])
AE = ([126.14,36.81,96.07])
AD = ([57.27,32.78,88.50])
AH = ([93.05,4.30,93.03])
AJ = ([82.38,83.83,9.83])
angles = np.array([AG,AK,AE,AD,AJ,AH])
angles = np.cos(angles) # convert all angles to their cosines
#print(angles)

'''
print("ENTER 11 COORDINATES: ")
O = get_array() 
A = get_array() 
B = get_array() 
Q = get_array() 
D = get_array() 
E = get_array() 
K = get_array() 
G = get_array() 
H = get_array() 
I = get_array() 
J = get_array()  
'''
coordinates = O, A, B, Q, D, E, K, G, H, I, J

# plot diagrams
fig = plt.figure(figsize=(15, 15/3))
ax0, ax1, ax2 = fig.subplots(ncols=3)

plot_vector(K, A, 'r', '$R_K$')
plot_vector(G, A, 'r', '$R_G$')
plot_vector(D, B, 'b', '$R_D$')
plot_vector(E, B, 'b', '$R_E$')
plot_vector(H, Q, 'g', '$R_H$')
plot_vector(J, I, 'purple', '$R_J$')

# calculate force magnitudes
forces,legends = susp_forces(coordinates, force_vector) 
#print(forces)

# calculate resolved forces using ( Force Along Axis = Magnitude of Force * Cos(Angle of Vector w.r.t Axis) )
FG = forces[0] * angles[0]
FK = forces[1] * angles[1]
FE = forces[2] * angles[2]
FD = forces[3] * angles[3]
FJ = forces[4] * angles[4]
FH = forces[5] * angles[5]

# create an array of the resolved forces
resolved_forces = np.array([FG, FK, FE, FD, FJ, FH])

# add legends for easy comprehension
final = np.concatenate((resolved_forces,legends),axis=1)

# print output
print(final)
fig.show()

