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

    # np.savetxt('text.txt', data)
    
    max_iter = 20

    # construct the inital curve
    row, col = image.shape

    x, y = np.arange(0, row), np.arange(0, col)

    X, Y = np.meshgrid(x, y)
    cell_size = np.pi / 10
    init_curve = cell_size * (np.sin(X) + np.sin(Y))

    # init_curve = np.ones((row,col))
    # init_curve[185:285, 140:340] *= -1

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
    eps_delta = 1e-1
    # coef for heaviside function
    eps_heaviside = 1e-1

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
        curvature = (phi_xx * phi_y**2 + phi_yy * phi_x**2 - 2*phi_xy * phi_y*phi_x)
        curvature[(phi_x!=0)&(phi_y!=0)] /= ((phi_x[(phi_x!=0)&(phi_y!=0)]**2 + phi_y[(phi_x!=0)&(phi_y!=0)]**2)**(3/2))

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

        # update
        # determind the average of each region
        c1 = np.sum(data[phi>0]*edge_constraint[phi>0])/(np.sum(edge_constraint[phi>0]))
        c2 = np.sum(data[phi<0]*edge_constraint[phi<0])/(np.sum(edge_constraint[phi<0]))

        # if np.sqrt(((phi-prev_phi)**2).mean()) < tol:
            # break
        if t%10==0:
            plt.figure(f'{t} on image')
            plt.imshow(image, cmap = plt.cm.bone)
            plt.contour(phi, [0], colors='r', linewidths=0.5)
            plt.title('t contour on image')
            print(f'iter: {t}')
            print(f'error: {np.sqrt(((phi-prev_phi)**2).mean())}')

        
    # plot the result
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(image)
    ax[0,0].contour(init_curve, [0], colors='r', linewidths=0.5)
    ax[0,0].set_title('Original image with initial curve')
    ax[0,0].axis('off')

    ax[0,1].imshow(image, cmap = plt.cm.bone)
    ax[0,1].contour(phi, [0], colors='r', linewidths=0.5)
    ax[0,1].set_title(f'After {max_iter} iteration')
    ax[0,1].axis('off')

    ax[1,0].imshow((phi>0).astype(int)*1, cmap = plt.cm.bone)
    ax[1,0].set_title('region1')
    ax[1,0].axis('off')

    ax[1,1].imshow((phi<0).astype(int)*1, cmap = plt.cm.bone)
    ax[1,1].set_title('region2')
    ax[1,1].axis('off')

    # plt.figure('Energy')
    # plt.plot( energy_storage, '*-')
    # plt.ylabel('energy')
    # plt.xlabel('time') 
    # plt.title('Energy at each time')


    # Plot the surface.
    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, phi, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()
    return 0


if __name__ == "__main__":
    a = LuBZ('./src/1')
