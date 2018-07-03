import glob
import time
import matplotlib.image as mpimg
import cv2
from features import get_hog_features, extract_features
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np



def compare_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def show_images(image1, image2, image3, image4,  image1_exp="Image 1", image2_exp="Image 2", image3_exp="Image 3", image4_exp="Image 4"):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    ax3.imshow(image3)
    ax3.set_title(image3_exp, fontsize=20)
    ax4.imshow(image4)
    ax4.set_title(image4_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def training(images, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    # Get image file names
    # images = glob.glob('./training-data/*/*/*.png')
    cars = []
    notcars = []
    all_cars = []
    all_notcars = []

    for image in images:
        if 'nonvehicle' in image:
            all_notcars.append(image)
        else:
            all_cars.append(image)

    # Get only 1/5 of the training data to avoid overfitting
    for ix, notcar in enumerate(all_notcars):
        # if ix % 5 == 0:
        notcars.append(notcar)

    for ix, car in enumerate(all_cars):
        # if ix % 5 == 0:
        cars.append(car)


    car_image = mpimg.imread(cars[5])
    notcar_image = mpimg.imread(notcars[0])
    compare_images(car_image, notcar_image, "Car", "Not Car")

    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"


    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    converted_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
    car_ch1 = converted_car_image[:,:,0]
    car_ch2 = converted_car_image[:,:,1]
    car_ch3 = converted_car_image[:,:,2]

    converted_notcar_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YUV)
    notcar_ch1 = converted_notcar_image[:,:,0]
    notcar_ch2 = converted_notcar_image[:,:,1]
    notcar_ch3 = converted_notcar_image[:,:,2]

    car_hog_feature, car_hog_image = get_hog_features(car_ch1,
                                            orient, pix_per_cell, cell_per_block,
                                            vis=True, feature_vec=True)

    notcar_hog_feature, notcar_hog_image = get_hog_features(notcar_ch1,
                                            orient, pix_per_cell, cell_per_block,
                                            vis=True, feature_vec=True)

    car_ch1_features = cv2.resize(car_ch1, spatial_size)
    car_ch2_features = cv2.resize(car_ch2, spatial_size)
    car_ch3_features = cv2.resize(car_ch3, spatial_size)
    notcar_ch1_features = cv2.resize(notcar_ch1, spatial_size)
    notcar_ch2_features = cv2.resize(notcar_ch2, spatial_size)
    notcar_ch3_features = cv2.resize(notcar_ch3, spatial_size)

    show_images(car_ch1, car_hog_image, notcar_ch1, notcar_hog_image, "Car ch 1", "Car ch 1 HOG", "Not Car ch 1", "Not Car ch 1 HOG")
    show_images(car_ch1, car_ch1_features, notcar_ch1, notcar_ch1_features, "Car ch 1", "Car ch 1 features", "Not Car ch 1", "Not Car ch 1 features")
    show_images(car_ch2, car_ch2_features, notcar_ch2, notcar_ch2_features, "Car ch 2", "Car ch 2 features", "Not Car ch 2", "Not Car ch 2 features")
    show_images(car_ch3, car_ch3_features, notcar_ch3, notcar_ch3_features, "Car ch 3", "Car ch 3 features", "Not Car ch 3", "Not Car ch 3 features")



    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    return svc, X_scaler