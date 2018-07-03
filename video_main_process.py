import imageio
import glob
imageio.plugins.ffmpeg.download()
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from collections import deque
from training import training
from sliding_window import apply_sliding_window
from heapmaps import add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label

def detect_cars(image):

    output_image, bboxes = apply_sliding_window(image, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                                spatial_size, hist_bins)

    heat = np.zeros_like(output_image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, bboxes)

    # Apply threshold to help remove false positives
    threshold = 1
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    current_heatmap = np.clip(heat, 0, 255)
    history.append(current_heatmap)

    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in history:
        heatmap = heatmap + heat

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img



if __name__ == '__main__':
    global history, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins

    history = deque(maxlen = 8)

    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    orient = 15  # HOG orientations

    images = glob.glob('./training-data/*/*/*.png')
    print(images)
    svc, X_scaler = training(images, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    image = mpimg.imread('./test_images/test4.jpg')
    img = detect_cars(image)
    fig = plt.gcf()
    fig.set_size_inches(16.5, 8.5)
    plt.imshow(img)
    plt.show()

    output = 'project_video_result.mp4'
    clip = VideoFileClip("project_video.mp4")
    video_clip = clip.fl_image(detect_cars)
    video_clip.write_videofile(output, audio=False)