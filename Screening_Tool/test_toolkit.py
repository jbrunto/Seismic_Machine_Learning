from turtle import shape
from seismic_toolkit import *
import numpy as np

def test_load_segy():
    test_path_1 = './Samson_cropped.segy'
    test_path_2 = './FWI_velocity_crop.segy'

    dummy_1 = load_segy(test_path_1, verbose=False)
    dummy_2 = load_segy(test_path_2, verbose=False)

    assert dummy_1.shape == dummy_2.shape

def test_standardize():

    test_data = np.load('./FWI_100.npy')
    test_data = test_data[0:1000, :, :]
    dummy_results = standardize(test_data, reshape=True)
    assert -1e-3 < np.mean(dummy_results) < 1e-3
    assert 1 - 1e-3 < np.std(dummy_results) < 1 + 1e-3

    dummy_results_2 = standardize(test_data, reshape=False)
    assert dummy_results_2.shape == test_data.shape
    assert dummy_results.shape[1] == dummy_results_2.shape[1] * dummy_results_2.shape[2]
    assert dummy_results.shape[0] == dummy_results_2.shape[0]


def test_patch():

    test_data = np.load('./marfurt_unpatched.npy')
    image_shape = [50, 50]
    test_data = test_data[:, :, 0:20]

    results_1 = patch(test_data, shape=image_shape)

    assert results_1.shape[1] == image_shape[0]
    assert results_1.shape[2] == image_shape[1]

    results_2 = patch(test_data)

    assert results_2.shape[1] == 100
    assert results_2.shape[2] == 100
    assert results_2.shape[0] != results_1.shape[0]


def test_unpatch():
    test_data = np.load('./marfurt_100.npy')
    test_data = test_data[0:900, :, :]
    image_shape = (9, 600, 1500)
    
    results = unpatch(test_data, image_size=image_shape)

    assert results.shape == image_shape


def test_cluster():
    test_data = np.load('./marfurt_100.npy')
    test_data = test_data[0, :, :].reshape((10000, 1))

    dummy_Inert = cluster(test_data, output='Inert')
    assert type(dummy_Inert) == float

    dummy_mask = cluster(test_data, output='Mask')
    assert type(dummy_mask) == np.ndarray
    assert dummy_mask.shape == test_data.shape

    dummy_all = cluster(test_data, output='All')
    assert len(dummy_all) == 4

def test_score_inertia():
    test_data = np.load('./marfurt_100.npy')
    test_data = test_data[0:30, :, :].reshape((30, 10000, 1))
    clusters = [2, 3, 4, 5]
    results = score_inertia(test_data, n_clusters=clusters)

    assert results.shape[0] == test_data.shape[0]
    assert results.shape[1] == len(clusters)


def test_score_var():
    test_data = np.load('./marfurt_100.npy')
    test_data = test_data[0:10, :, :].reshape((10, 10000, 1))
    clusters = [2, 3, 4, 5]
    results = score_var(test_data, n_clusters=clusters)

    assert results.shape[0] == test_data.shape[0]
    assert results.shape[1] == len(clusters)

def test_predict():
    test_data = np.load('./marfurt_100.npy')
    test_data = test_data[0:30, :, :].reshape((30, 10000, 1))
    results_masks, results_inerts = predict(test_data, verbose=False)
    assert results_masks.shape[1] == results_inerts.shape[0]
    assert results_masks.shape[1] == test_data.shape[0]
    assert results_masks.shape[0] == test_data.shape[1]

def test_mask():
    test_data_masks = np.load('./train_masks.npy')
    test_data_velos = np.load('./FWI_patched_train.npy')
    
    results_1, results_2 = mask(test_data_velos, test_data_masks)
    assert results_1.shape[0] == test_data_velos.shape[0]
    assert results_1.shape[1] == test_data_velos.shape[1] * test_data_velos.shape[2]
    assert results_1.shape == results_2.shape

def test_marfurt():
    test_data = load_segy('./Samson_cropped.segy', verbose=False)
    test_data = test_data[0:300, 0:1000, 0:25]
    dummy = test_data[:, :, 0]

    results = marfurt(test_data)
    
    assert results.shape == test_data.shape
    assert results[:, :, 0].all() == dummy.all()

def test_elbow_inertia():
    test_data = np.load('./scores_5_45.npy')
    n_clusters = [5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20,
        25, 22, 27, 30]
    result = elbow_inertia(test_data[:, 0:15], verbose=False)

    assert type(result) == int
    assert result in n_clusters

def test_elbow_variance():
    test_data = np.load('./test_set_variance.npy')
    n_clusters = [5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 20,
        25, 22, 27, 30]
    result = elbow_variance(test_data[:, 0:15], verbose=False)

    assert type(result) == int
    assert result in n_clusters


