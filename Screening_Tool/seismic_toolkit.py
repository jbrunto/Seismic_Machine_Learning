import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from patchify import patchify, unpatchify
import segyio
import itertools
from mayavi import mlab
import scipy.ndimage


def load_segy(path, verbose=True):
    """
    Loads a .segy file into a usable data object (NumPy Array).

    Arguments.
        - path (str): A string that leads to the desired .segy file
        - verbose (Boolean): When True, displays .segy file metadata.
        Default = True.
    
    Returns.
        - cube (NumPy Array): A 3D volume retrieved from the .segy file, with
        dimensions (Ilines, Xlines, Samples/Depth)
    """
    # load the data using segyio
    data = segyio.open(path, ignore_geometry=False)
    if verbose == True:
        print(segyio.tools.wrap(data.text[0]))
    else:
        pass
    # Convert the data from segyio object to a NumPy Array
    cube = segyio.tools.cube(data)
    # Return Data
    return cube

def standardize(ims, reshape=True):
    """
    A function developed for doing basic preprocessing of the large
    data arrays used throughout each notebook. This function is essentially a
    wrapper around Sci-Kit Learn's StandardScaler() tool, but also serves to
    reshape the data into flattened, single channel images, an array shape
    required by the K-Means algorithm.

    Arguments.
        - ims (Numpy Array): a set of seismic images with shape (number of images,
         image width, image length).
        - reshape (Boolean): determines whether the data is reshaped to the
                dimensions required by the K-Means algorithm. The default function
                performs this reshaping.

    Returns.
        - a numpy array of shape according to the 'reshape' argument. 
    """
    ims_shape = ims.shape
    # flatten the 3D array for standardization
    ims = ims.flatten().reshape(ims.shape[0], ims.shape[1]*ims.shape[2])
    # instantiate the Standard Scaler
    ims_scaler = StandardScaler()
    # fit and transform the images
    ims_norm = ims_scaler.fit_transform(ims[:, :])
  
    # DEFAULT: return a flattened array of single channel images
    if reshape == True:
        return ims_norm.reshape(ims_norm.shape[0], ims_norm.shape[1], 1)
    # Do not change original shape if selected
    elif reshape == False:
        return ims_norm.reshape(ims_shape)

def patch(volume, shape = [100, 100]):
    """
    Converts Images from their original size to 100x100 pixel images that are
    a usable size for the K-Means clustering algorithm.

    Arguments.
        - volume (NumPy Array): a set of images to be patched. Dimensions of
        (Ilines, Xlines, Samples/Depth)
        - shape (list): a list object containing two integers, the desired image
        length and desired image width, respectively.
    
    Returns.
        - data (Numpy Array): The patched image dataset. Dimensions of (Number 
        of Samples, 100, 100).
    """
    for i in range(volume.shape[2]):
        patches = patchify(volume[:,:,i], shape, step=shape[0])
        patches = patches.reshape(patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3])
        if i == 0:
            data = patches.copy()
        else:
            data = np.concatenate((data, patches), axis=0)
    return data

def unpatch(patches, image_size=None):
    """
    The inverse operation of the patch() function above, converts a patched dataset
    back to the dimensions to the original volume. The number of samples (depth slices)
    remains the same, but the original images may naturally lose data along the edges if
    the volume is not a perfect rectangle.

    Arguments.
        - patches (NumPy Array): A set of images to be converted back to the dimensions
        closest to the original datasets shape. 
        - image_size (tuple): Tuple of integers of the desired image size. The default
        value is none, as this must be a user-given argument.

    Returns.
        - reconstr (NumPy Array): The reconstructed set of images with dimensions
        of the user given shape above.

    """
    # create an empty array to store the reconstructed volume
    reconstr = np.empty(image_size)
    for k in range(reconstr.shape[0]):
        for i in range(15):
            for j in range(6):
                # assign images to original locations
                reconstr[k, j*100:(j+1)*100, i*100:(i+1)*100] = patches[j + (i*6) + (k * 90)].reshape((100,100))
    return reconstr

def cluster(im, k=12, eps=1e-4, iters=10, trys=20, output='All'):
    """
    A function for performing a single model run of KMeans. The
    implementation of KMeans used is that developed by Kington, J.
    (2011) and the OpenCV team. All arguments with default values,
    aside from output, have values optimized to the Samson Dome
    Dataset (see Notebook).

    Arguments.
        - im (Numpy Array): A single seismic image in the form of a NumPy array
        with shape (image width * image length, 1)
        - k (int): Number of clusters to segment the image into. Default=12.
        - eps (float): Relative tolerance of K-Means algorithm. Default=1e-4.
        - iters (int): Maximum number of iterations of K-Means algorithm.
        Default = 10.
        - trys (int): Number of times the K-Means model reinitializes its
        algorithm. Default = 20.
        -  Output (str): [Default = 'All', 'Inert', 'Mask']. 'All' returns the images
        predicted mask, segmented image, centroids, and image inertia. 'Inert'
        returns predicted segmented image inertia only, and 'Mask' returns only the 
        predicted image mask.
    
    Returns.
        - mask (Optional, NumPy Array): Predicted flattened image mask, with clusters being
        assigned to the image background (value=0) or image foreground (value=1).
        - labels (Optional, NumPy Array): Predicted flattened segmented image, with
        pixels being assigned values according to predicted value.
        - centers (Optional, tuple): A tuple of predicted cluster centroids.
        - Inertia (Optional, float): Average intra-cluster sum of squared distances for each
        pixel. 
    """
    # define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, iters, eps)
    # convert array values to type np float32
    im = np.float32(im)
    # perform K-Means clustering
    inertia, labels, (centers) = cv.kmeans(im, k, None, criteria,
                                         trys, cv.KMEANS_RANDOM_CENTERS)
    # creating the mask from the returned cluster labels and centers
    centers = np.uint8(centers)
    mask=centers[labels.flatten()]
    
    if output == 'All':
        return mask, labels, centers, inertia
    elif output == 'Inert':
        return inertia
    elif output == 'Mask':
        return mask
    else:
        print('ValueError: Output not recognized')

def score_inertia(ims, n_clusters=[5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20,
                    22, 25, 27, 30], verbose=False):
    """
    A function that performs the initial part of the elbow-method, based
    on inertia. Scores the KMeans model predictions, and returns the average image
    score for each number of clusters specified.

    Arguments.
        - ims (NumPy Array): Set of flattened seismic images of shape (number of
        images, image length * image width, 1).
        - n_clusters (List, dtype=int): cluster values to be scored. Default list
        includes a range of values between 5 and 30.
        - verbose (Boolean): When True, prints milestone updates of the scoring
        through the dataset. Default = False.

    Returns.
        - scores_array (NumPy Array): An array with dimensions (Number of images,
         Number of clusters). Contains float values of average best inertia score
         for that image for each cluster value.
    """

    # loop through image set
    for i in range(ims.shape[0]):
        # print percent complete if verbose is true
        if i in [int(ims.shape[0]*0.1), int(ims.shape[0]*0.2), int(ims.shape[0]*0.3),
            int(ims.shape[0]*0.4), int(ims.shape[0]*0.5), int(ims.shape[0]*0.6),
            int(ims.shape[0]*0.7), int(ims.shape[0]*0.8),
            int(ims.shape[0]*0.9)] and verbose == True:
            print(str((i/ims.shape[0])*100), 'percent completed' )
        # instantiate the list of average intertia values for each cluster value
        averages = []
        # loop through each cluster value
        for k in n_clusters:
            # instantiate the list of scores so average can be calculated
            scores = []
            # Perform 5 runs to determine average best value
            for j in range(5):
                # get inertia score for image, cluster value pair
                inertia = cluster(ims[i, :, :], k=k, output='Inert')
                # store inertia score
                scores.append(inertia)
            # store average of best inertia scores
            averages.append(sum(scores)/len(scores))
        # create an array of the scores for each image
        scores = np.array(averages).reshape((1, len(n_clusters)))
        # if scores_array does not exist, simply create a copy of the array
        if i == 0:
            scores_array = np.copy(scores)
        # concatenate scores
        else:
            scores_array = np.concatenate((scores_array, scores), axis=0)
    # return scores  
    return scores_array

def score_var(ims, n_clusters = [5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20,
        25, 22, 27, 30], verbose=False):
    """
    A function that performs the initial part of the elbow-method, based
    on explained variance. Scores the KMeans model predictions, and returns
    the silhouette score for each number of clusters specified.
    Arguments.
        - ims (NumPy Array): Set of flattened seismic images of shape (number of
        images, image length * image width, 1).
        - n_clusters (List, dtype=int): cluster values to be scored. Default list
        includes a range of values between 5 and 30.
        - verbose (Boolean): When True, prints milestone updates of the scoring
        through the dataset. Default = False.

    Returns.
        - scores_array (NumPy Array): An array with dimensions (Number of images,
         Number of clusters). Contains float values of the silhouette score
         for that image for each cluster value.
    """
    # loop through image set
    for i in range(ims.shape[0]):
        # print percent complete if verbose is true
        if i in [int(ims.shape[0]*0.1), int(ims.shape[0]*0.2), int(ims.shape[0]*0.3),
            int(ims.shape[0]*0.4), int(ims.shape[0]*0.5), int(ims.shape[0]*0.6),
            int(ims.shape[0]*0.7), int(ims.shape[0]*0.8),
            int(ims.shape[0]*0.9)] and verbose == True:
            print(str((i/ims.shape[0])*100), 'percent completed' )
        # instantiate scores list- to be converted to np array
        scores_list = []
        # loop through each cluster value
        for k in n_clusters:
            # perform KMeans - Sci-Kit Learn Implementation
            model = KMeans(n_clusters=k, max_iter=10,
                            init='k-means++', n_init=20,
                            tol=1e-4)
            model.fit_predict(ims[i, :, :])
            # append score
            scores_list.append(silhouette_score(ims[i, :, :], model.labels_, metric='euclidean'))
        # convert from list to Numpy array
        scores = np.array(scores_list).reshape((1, len(n_clusters)))
        # if scores_array does not exist, simply create a copy of the array
        if i == 0:
            scores_array = np.copy(scores)
        # concatenate array if else
        else:
            scores_array = np.concatenate((scores_array, scores), axis=0)
    # return scores
    return scores_array

def predict(ims, verbose=True, k=12):
    """
    Performs a final prediction of the fault mask using the optimized K-Means model, while
    also recording the predicted image inertia.
    
    
    Arguments.
        - ims (NumPy Array): the dataset of images to have fault locations predicted by the
        K-Means model. Dimensions (# of images, flattened image length, 1)
        - verbose (Boolean): When True, prints milestone updates of the function.
        
    Returns.
        - masks (NumPy Array): the binary masks obtained from K-Means segmentation.
        Dimensions
        of (# of images, flattened image length).
        - inerts (NumPy Array): the inertia of each predicted image. Dimensions of (# of images, 1).
    """
    inerts = np.empty(shape=(ims.shape[0]))
    for i, j in enumerate(ims):
        mask, _, _, inert  = cluster(j, k=k)
        if i == 0:
            masks = mask.copy()
        else:
            masks = np.concatenate((masks, mask), axis=1)
        inerts[i] = inert
        # print percent complete if verbose is true
        if i in [int(ims.shape[0]*0.1), int(ims.shape[0]*0.2), int(ims.shape[0]*0.3),
            int(ims.shape[0]*0.4), int(ims.shape[0]*0.5), int(ims.shape[0]*0.6),
            int(ims.shape[0]*0.7), int(ims.shape[0]*0.8),
            int(ims.shape[0]*0.9)] and verbose == True:
            print(str((i/ims.shape[0])*100), 'percent completed' )
    
    return masks, inerts

def mask(velos, masks):
    """
    Combines a volume of predicted fault locations with FWI velocity data to create
    a single inteprative volume.

    Arguments.
        - velos (NumPy Array): FWI-recovered velocity dataset of dimensions (# of
        seismic images, image width, image length).
        - masks (NumPy Array): K-Means predicted fault locations dataset of dimensions
        (image length * image width, # of seismic images).

    Returns.
        - results_in (NumPy Array): The combined dataset of flattened images with
        dimensions (# of seismic images, image length * image width).
    """
    # Check the dimensions of the datasets are appropriate
    assert velos.shape[0] == masks.shape[1]
    # Create empty numpy array of desired size
    results_in = np.empty((velos.shape[0], velos.shape[1]*velos.shape[2]))
    results_out = np.empty((velos.shape[0], velos.shape[1]*velos.shape[2]))
    # Loop through each image to combine
    for i in range(velos.shape[0]):
        image = velos[i, :, :]
        # reshape mask and standardize
        fault_mask = masks[:, i].reshape((velos.shape[1], velos.shape[2])) / 255
        nonfault_mask = fault_mask + 1
        nonfault_mask[nonfault_mask > 1] = 0
        # multiply the FWI velocity data by the fault mask to extract info.
        result_in = (image * fault_mask).astype(np.uint8)
        result_out = (image * nonfault_mask).astype(np.uint8)
        results_in[i, :] = result_in.reshape((velos.shape[1]*velos.shape[2]))
        results_out[i, :] = result_out.reshape((velos.shape[1]*velos.shape[2]))
    # return complete array
    return results_in, results_out

def explore3d(data_cube, preset = True, I=-1, X = -1 , Z=-1):
    """
    Visualizes large 3D data volumes efficiently with Mayavi, and allows the user
    to interpret the dataset through interactive zooming and scrolling.

    Arguments.
        - data_cube (NumPy Array): A 3D volume to be visualized. The dimensions of
        this array should be the closest possible match to the original volume
        shape, immediately following the unpatch() function.
        - preset (Boolean): When True, determines the initial slices displayed
        based on the below function
        - I (int): Initial Iline slice displayed. Default = -1.
        - X (int): Initial Xline slice displayed. Default = -1.
        - Z (int): Initial Depth slice displayed. Default = -1.

    Returns.
        - None. Display window must be closed before peforming any further operations
        within the notebook.
    """
    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]
    vm = np.percentile(data_cube, 95) #may need to play a little with the 95
    
    if preset == True:
        nx, ny, nz = data_cube.shape
        I = nx//2
        X = ny//2
        Z = nz//2

    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes', 
                                     slice_index=I, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes', 
                                     slice_index=X, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes', 
                                     slice_index=Z, colormap='coolwarm', vmin=-vm, vmax=vm)
    mlab.show()


def moving_window(data, window, func):
    """
    The moving window portion of Kington, J.'s implementation of marfurt 
    semblance-based coherency seismic attribute creation. For the full repository,
    see https://github.com/seg/tutorials-2015/blob/master/1512_Semblance_coherence_and_discontinuity/writeup.md .
    All comments given are
    Arguments.
        - data (NumPy Array): the 3D seismic dataset of which to extract the
        seismic attribute of. Dimensions of (Ilines, Xlines, Samples/Depth).
        - window (List): List of integers that describe the boundaries of the window
        that determines an observation's value. A larger area is more computationally
        expensive, but describes each observation more accurately.
        - func (Python function): the function that calculates the desired attribute

    Returns.
        - The newly calculated 3D seismic attribute with dimensions equal to that of
        the original dataset used.
    """
    # `generic_filter` will give the function 1D input and reshape
    wrapped = lambda region: func(region.reshape(window))
    
    # scipy function to apply to data
    return scipy.ndimage.generic_filter(data, wrapped, window)

def marfurt_semblance(region):
    """
    Marfurt semblance as implemented by Kington, J. (2016). See moving_window()
    for link to description. A function that determines an observation's value
    based on the values of the pixels in the neighbouring region.

    Arguments.
        region (NumPy Array): In this implementation, is specified by the window
        argument of the moving_window() function above. 

    Returns.
        A value assigned to an individual observation.
    """
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape

    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces


def marfurt(volume):
    """
    A function that wraps the two functions from Kington, J. (2016) above,
    and determines the region of interest when calculating the marfurt semblance
    attribute.

    Arguments.
        - volume (NumPy Array): 3D seismic image dataset to extract the
        Marfurt semblance-based coherency seismic attribute from, should be
        the dimensions of the original volume.

    Returns.
        - A NumPy Array of the same dimensions as the volume argument.
    """
    return moving_window(volume, [3,3,9], marfurt_semblance)

def elbow_inertia(scores, n_clusters = [5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20, 25, 22, 27, 30], verbose=True):
    """
    Visualizes the 'elbow-method' (see Reports) for inertia with a MatPlotLib figure. Also, gives a mathematical 
    interpretation of the elbow-method.
    
    Arguments.
        - scores (NumPy Array): An array containing the scores for each number of clusters for each image.
        Dimensions of (# of seismic images, # of clusters)
        - n_clusters (List of ints.): A list of integers containing the number of clusters to be evaluated
        using the K-Means algorithm.
        - verbose (Boolean): When True, displays the mathematical interpretation of the elbow-method, or the
        predicted optimal value for number of clusters for this K-Means model for this set of images.
        
    Returns.
        - best_k (int): The mathematical interpretation for the optimal value of k (number of clusters). It
        is derived by maximizing the distance between the average trend in inertia and observed values for
        inertia (see below and Report for more).
    
    """
    # create a figure object
    fig = plt.figure(figsize=(15,10))
    # calculate the average inertia for each cluster value across the dataset
    model_avgs = [np.sum(scores[:, i])/scores.shape[0] for i in range(len(n_clusters))]
    # plot the model averages calculated above against the cluster values, k
    plt.scatter(n_clusters, model_avgs)
    # fit and plot (z & p) a trendline for the inertia
    z = np.polyfit(n_clusters, model_avgs, 10)
    p = np.poly1d(z)
    plt.plot(n_clusters, p(n_clusters), 'r')
    # plot the average trend in inertia
    plt.plot([n_clusters[0], n_clusters[-1]], [model_avgs[0], model_avgs[-1]], 'g')
    m, b = np.polyfit([n_clusters[0], n_clusters[-1]], [model_avgs[0], model_avgs[-1]], 1)
    # create in-built functions for calculating points and distances
    def y(x):
        return (m*x) + b
    def x(y):
        return (y - b)/m
    dists = []
    # calculate vertical, horizontal, and total distance, and append total distance
    for i in range(len(n_clusters)):
        vert_dist = abs(model_avgs[i] - y(n_clusters[i]))
        hor_dist = abs(n_clusters[i] - x(model_avgs[i]))
        tot_dist = ((vert_dist**2)+(hor_dist**2))**0.5
        dists.append(tot_dist)
    # plot the difference between the average trendline and the fitted trendline
    plt.plot(n_clusters, dists, 'b')
    # obtain best value for number of clusters, k and print to screen if verbose
    # is True, always return best_k
    high_dist = max(dists)
    best_k_index = dists.index(high_dist)
    best_k = n_clusters[best_k_index]
    if verbose == True:
        print('The optimal choice for K is: ', best_k)
    return best_k

def elbow_variance(scores, n_clusters = [5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20, 25, 22, 27, 30], verbose=True):
    """
    Visualizes the 'elbow-method' according to variance. The elbow method for variance is simpler than that for
    inertia, as the maximum model average is considered to be the optimal choice.
    
   Arguments.
        - scores (NumPy Array): An array containing the scores for each number of clusters for each image.
        Dimensions of (# of seismic images, # of clusters)
        - n_clusters (List of ints.): A list of integers containing the number of clusters to be evaluated
        using the K-Means algorithm.
        - verbose (Boolean): When True, displays the mathematical interpretation of the elbow-method, or the
        predicted optimal value for number of clusters for this K-Means model for this set of images.
        
    Returns.
        - best_k (int): The mathematical interpretation for the optimal value of k (number of clusters). It
          is simply the maximum model average. 
    """
    # create matplotlib figure to be displayed
    fig = plt.figure(figsize=(15,10))
    # compute average silhouette score for each value for number of clusters, k
    model_avgs = [np.sum(scores[:, i])/scores.shape[0] for i in range(len(n_clusters))]
    # display points and fit a trend line
    plt.scatter(n_clusters, model_avgs)
    z = np.polyfit(n_clusters, model_avgs, 10)
    p = np.poly1d(z)
    plt.plot(n_clusters, p(n_clusters), 'r', label='fitted trend')
    plt.legend(loc='best')
    # obtain optimal k value and display best_k if verbose is True
    best_k_index = model_avgs.index(max(model_avgs))
    best_k = n_clusters[best_k_index]
    if verbose == True:
        print('The optimal choice for K is :', best_k)
    return best_k
