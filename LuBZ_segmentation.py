import numpy as np
import matplotlib.pyplot as plt
import pydicom
import time

# main function
def LuBZ(file_path):
    file = pydicom.read_file(file_path)
    image = file.pixel_array

    # normlize
    data = image/(np.max(image)-np.min(image))
    
    max_iter = 100

    # construct the inital curve
    row, col = image.shape

    x, y = np.arange(0, row), np.arange(0, col)

    X, Y = np.meshgrid(x, y)
    cell_size = np.pi / 5
    init_curve = cell_size * (np.sin(X) + np.sin(Y))
    # init_curve = np.ones((row,col))*-1
    # init_curve[200-30:150+30,200-30:150+30] *= -1
    
    zero_level_width = 1e-3 # determind the contour width
    phi = init_curve
    phi = phi/(np.max(phi)-np.min(phi))
    phi[abs(phi) < zero_level_width] = 0 # narrow the cell

    gradient_square = np.gradient(data)[1]**2+np.gradient(data)[0]**2

    # determind the average of each region
    c1 = np.mean(data[phi > 0])
    c2 = np.mean(data[phi < 0])

    # the constraint of dragging force
    edge_constraint = 1 / (1+gradient_square)

    # coef for delta function
    eps_delta = 1e-3
    # coef for heaviside function
    eps_heaviside = 1e-3

    # coef force term
    lambda1 = 1
    lambda2 = 1

    # coef zero level set
    mu = 1e-3

    # time step. 0 < time_step <=0.25 see [1, (27)]
    time_step = 0.1

    # storage things
    energy_storage = []

    # stop criterion
    tol = 1e-3

    for t in range(0, max_iter):
        prev_phi = phi

        # compute the force of each region [1, (25)]
        # the force also could be considered as dissimilarity function in [1, (18)]
        difference_from_average1 = (data - c1)**2
        difference_from_average2 = (data - c2)**2
        force_denominator =  2 + (difference_from_average1 + difference_from_average2)*2*edge_constraint
        force1 = (1 + 2*edge_constraint*difference_from_average1) / force_denominator
        force2 = (1 + 2*edge_constraint*difference_from_average2) / force_denominator
        
        # compute the curvature
        phi_x = np.gradient(prev_phi)[1]
        phi_y = np.gradient(prev_phi)[0]
        phi_xx = np.gradient(phi_x)[1]
        phi_yy = np.gradient(phi_y)[0]
        phi_xy = np.gradient(phi_x)[0]
        curvature = (phi_xx * phi_y**2 + phi_yy * phi_x**2 - 2*phi_xy * phi_y*phi_x) / ((phi_x**2 + phi_y**2)**(3/2))

        # Dirac delta
        dirac_delta = eps_delta/np.pi/(prev_phi**2+eps_delta**2)

        # Heaviside
        heaviside = 1/2 + np.arctan(prev_phi/eps_heaviside)/np.pi

        gradient_phi = np.gradient(phi)
        energy_data = np.sum(lambda1*force1*heaviside+lambda2*force2*(1-heaviside))
        energy_length = np.sum(mu*dirac_delta*(gradient_phi[1]**2+gradient_phi[0]**2)**0.5)
        energy_storage.append(energy_data+energy_length)

        # evolution
        # backward for finite difference in time. time step is one [1, (26)]
        phi_half = prev_phi - dirac_delta*(lambda1*force1-lambda2*force2-mu*curvature)
        
        phi_x = np.gradient(phi_half)[1]
        phi_y = np.gradient(phi_half)[0]
        phi_xx = np.gradient(phi_x)[1]
        phi_yy = np.gradient(phi_y)[0]
        laplacian = phi_xx + phi_yy

        # full time [1, (27)]
        phi = phi_half + time_step * laplacian
        phi[abs(phi) < zero_level_width] = 0 # narrow the cell

        # update
        # determind the average of each region
        c1 = np.sum(data[phi>0]*edge_constraint[phi>0])/np.sum(edge_constraint[phi>0])
        c2 = np.sum(data[phi<0]*edge_constraint[phi<0])/np.sum(edge_constraint[phi<0])

        # if np.sqrt(((phi-prev_phi)**2).mean()) < tol:
            # break
        # if t%20==0:
        #     plt.figure(f'{t} on image')
        #     plt.imshow(image, cmap = plt.cm.bone)
        #     plt.contour(phi, [0], colors='r', linewidths=0.5)
        #     plt.title('t contour on image')
        #     print(f'iter: {t}')
        #     print(f'error: {np.sqrt(((phi-prev_phi)**2).mean())}')

        
    # plot the result
    plt.figure('initial image')
    plt.imshow(image)
    plt.title('Original image')

    plt.figure('contour on image')
    plt.imshow(image, cmap = plt.cm.bone)
    plt.contour(phi, [0], colors='r', linewidths=0.5)
    plt.title('Final contour on image')

    plt.figure('region1 on image')
    plt.imshow((phi>0).astype(int)*1, cmap = plt.cm.bone)
    plt.title('region1')

    plt.figure('region2 on image')
    plt.imshow((phi<0).astype(int)*1, cmap = plt.cm.bone)
    plt.title('region2')

    plt.figure('Energy')
    plt.plot( energy_storage, '*-')
    plt.ylabel('energy')
    plt.xlabel('time') 
    plt.title('Energy at each time')

    plt.show()
    return 0


if __name__ == "__main__":
    a = LuBZ('./src/1')
