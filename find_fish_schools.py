import numpy as np
import echopype as ep
import pandas as pd
import glob
import os
import sys
import warnings
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import yaml
import traceback
from scipy.ndimage import median_filter
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from yaml.loader import SafeLoader
from find_bottom import find_bottom
from export_data import save_data
from find_fish import find_fish_median, medianfun
from visualization import data_to_images
from processing import process_data, clean_times, remove_vertical_lines
from find_waves import find_waves, find_layer

warnings.filterwarnings("ignore")


# Load all params from yaml-file
with open(sys.argv[5], 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))


# Remove already processed files  
path = sys.argv[1]
completed_files_path = sys.argv[2]
new_processed_files_path = sys.argv[3]
csv_path = sys.argv[4]
img_path = sys.argv[6]
gps_path = sys.argv[7]
sonar_depth = params[0]['sonar_depth']


files = os.listdir(path)

completed_txt_file = open(completed_files_path, 'r')
completed_files = [line for line in completed_txt_file.readlines()]
completed_files = [file.replace('\n', '') for file in completed_files]

files = [f for f in files if f not in completed_files]

open(new_processed_files_path, "w").close()

times = []
longitudes = []
latitudes = []
depths = []
widths = []
heights = []
areas = []
nasc_means = []
nasc_totals = []
intensities = []
densities = []
bottom_depths = []


df_gps = pd.read_csv(gps_path)
df_gps['Datetime'] = pd.to_datetime(df_gps['GPS_date'] + ' ' + df_gps['GPS_time'])
df_gps = df_gps.drop(['GPS_date', 'GPS_time'], axis=1)

if files:
    for file in files:
        print(file)
        if '.raw' in file:
            try: 
                with open(completed_files_path, 'a') as txt_doc:
                    txt_doc.write(f'{file}\n')

                filepath = f'{path}/{file}'
                new_file_name = filepath.split('/')[-1].replace('.raw', '')

                # Load and process the raw data files
                echodata, ping_times = process_data(filepath, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
                echodata = echodata.Sv.to_numpy()[0]
                echodata, nan_indicies = remove_vertical_lines(echodata)
                echodata_swap = np.swapaxes(echodata, 0, 1)

                data_to_images(echodata_swap, f'{img_path}/{new_file_name}') # save img without ground
                os.remove(f'{img_path}/{new_file_name}_greyscale.png')

                # Detect bottom algorithms
                depth, hardness, depth_roughness, new_echodata = find_bottom(echodata_swap, params[0]['move_avg_windowsize'])
                bottom_depth = depth 
                # Find, measure and remove waves in echodata
                new_echodatax = new_echodata.copy()
                layer = find_layer(new_echodatax, params[0]['beam_dead_zone'], params[0]['layer_in_a_row'], params[0]['layer_quantile'], params[0]['layer_strength_thresh'], params[0]['layer_size_thresh'])
                if layer:
                    new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])
                else:
                    new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params[0]['wave_thresh'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

                    if wave_avg > params[0]['extreme_wave_size']: 
                        new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodatax, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

                original_echodata = new_echodata.copy()
                data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete') # save img without ground and waves
                os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')


                #Creates binary image
                binary_echodata = np.where((new_echodata > -70) & (new_echodata < 0), 1, 0) #binary_echodata = np.where(new_echodata > -75, 1, 0)
                image = (binary_echodata * 255).astype(np.uint8)
                _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY) # använder thresholding för att göra alla 1or vita och resten svarta
                binary_image_copy = binary_image.copy()
                #plt.imsave(f'{img_path}/{new_file_name}_after_thresholding.png', binary_image)

                #blurres the image
                blur = cv2.blur(binary_image,(5,10))
                #plt.imsave(f'{img_path}/{new_file_name}_after_blurring.png', blur)

                #Finds contours
                contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #cv2.drawContours(blur, contours, -1, (255), 1)  # -1 means draw all contours, (0, 255, 0) is the color, 1 is the thickness
                #plt.imsave(f'{img_path}/{file_name}_contours.png', blur)

                convex_hulls = [cv2.convexHull(c) for c in contours]
                cv2.drawContours(blur, convex_hulls, -1, (255), 1)
                #plt.imsave(f'{img_path}/{new_file_name}_convex_hull.png', blur)

                for i, hull in enumerate(convex_hulls):

                    area = cv2.contourArea(hull)

                    black_frame = np.zeros_like(original_echodata).astype(np.uint8)
                    cv2.fillPoly(black_frame , [hull], (255, 255, 255))
                    mask = black_frame == 255
                    targetROI = original_echodata * mask

                    targetROI_filtered = np.where(targetROI > -70, targetROI, np.nan)
                    intensity = np.nanmean(targetROI_filtered)
        
                    
                    
                    black_frame = np.zeros_like(binary_image)
                    cv2.fillPoly(black_frame, [contours[i]], (255, 255, 255))
                    targetROI_binary = cv2.bitwise_and(binary_image, black_frame)
                    white_pixels_count = np.count_nonzero(targetROI_binary == 255)
                    total_pixels_count = cv2.contourArea(hull)
                    density = white_pixels_count/total_pixels_count
                    
                    # cv2.drawContours(binary_image_copy, [hull], 0, 255, 1)
                    # plt.imsave(f'{img_path}/{file_name}_binary_hull.png', binary_image_copy)

                    if density>0.2 and area > 200 and intensity>-0.7: #intensity>0.7 räcker för botten 
                        # for point in hull:
                        #     point[0][1] +=max(wave_line)
                        cv2.drawContours(original_echodata, [hull], 0, (0, 255, 0), 1)

                        #Extracts the centroid
                        M = cv2.moments(hull)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0

                        #Get time based on x coordinate of controid
                        time = ping_times[cX]

                        #Get depth 
                        depth = cY*0.1

                        #Find the closest time in df_gps to get coordinates
                        index_of_coordinate = np.argmin(np.abs(df_gps['Datetime'].values - time))
                        latitude = df_gps['Latitude'][index_of_coordinate]
                        longitude = df_gps['Longitude'][index_of_coordinate]

                        x, y, w, h = cv2.boundingRect(hull)

                        width = w   
                        height = h    
                        

                        nasc = 4 * np.pi * (1852**2) * (10**(targetROI/10)) * 0.1
                        nasc_total = np.sum(nasc)
                        nasc_mean = np.mean(nasc)
                        
                        data_to_images(targetROI, f'{img_path}/{new_file_name}_target_ROI')
                        os.remove(f'{img_path}/{new_file_name}_target_ROI.png')    

                        data_to_images(nasc, f'{img_path}/{new_file_name}_nasc')
                        os.remove(f'{img_path}/{new_file_name}_nasc_greyscale.png')    

                        
                        data_to_images(original_echodata, f'{img_path}/{new_file_name}_clusters_on_original')
                        os.remove(f'{img_path}/{new_file_name}_clusters_on_original_greyscale.png')    


                        times.append(time)
                        latitudes.append(latitude)
                        longitudes.append(longitude)
                        depths.append(depth)
                        widths.append(width)
                        heights.append(height)
                        areas.append(area)
                        nasc_means.append(nasc_mean)
                        nasc_totals.append(nasc_total)
                        intensities.append(intensity)
                        densities.append(density)
                        bottom_depths.append(np.median(bottom_depth)*0.1)
        

            except KeyboardInterrupt:
                df = pd.DataFrame({
                'time': times,
                'longitude': longitudes,
                'latitude': latitudes,
                'depth': depths,
                'width': widths,
                'height': heights,
                'area': areas,
                'nasc_mean': nasc_means,
                'nasc_total': nasc_totals,
                'intensity': intensities,
                'density': densities
                })

                df.to_csv(csv_path, index=False)

            
            except Exception as error:
                    traceback.print_exc()
                    print(f'Problems with {new_file_name }')


for list in [longitudes, latitudes, depths, widths, heights, areas, nasc_means, nasc_totals, intensities, densities, bottom_depths]:
    list[:] = [round(x, 2) for x in list]



df = pd.DataFrame({
'time': times,
'longitude': longitudes,
'latitude': latitudes,
'depth': depths,
'width': widths,
'height': heights,
'area': areas,
'nasc_mean': nasc_means,
'nasc_total': nasc_totals,
'intensity': intensities,
'density': densities,
'bottom_depth': bottom_depths
})

print(df.head())

df.to_csv(csv_path, index=False)



        