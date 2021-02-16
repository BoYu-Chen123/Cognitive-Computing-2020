import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
from PIL import ImageFont, ImageDraw, Image
from numpy.linalg import norm
from sklearn.cluster import KMeans

#############################color feature parameter
color_slicing = 3
bgr_bins = [8, 8, 8]
hsv_bins = [18, 3, 3]
image_size = [256, 256]
#############################edge feature parameter
edge_slicing = 4
angel_slicing = 36
edge_dominate_thres = 12
#############################local feature parameter
cluster_nums = 80
nfeatures_less = 130
nfeatures_more = 280

def read_file(filename):
    categories = os.listdir(filename)
    categories.sort()
    namelist = []
    for category in categories:
        temp = []
        if category != ".DS_Store":
            temp.append(category)
            img_names = os.listdir(os.path.join(filename, category))
            img_names.sort(key=lambda x: int(re.split('[_.]', x)[-2]))
            for img_name in img_names:
                if img_name != ".DS_Store":
                    temp.append(img_name)
            namelist.append(temp)
    return namelist

########### color feature ###########
def color_hist(image):
#     plt.imshow(image)
    b, g, r = cv2.split(image)
    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsvimg)
    
    rows, cols = h.shape[0], h.shape[1]
    r_gsize = int(rows / color_slicing)
    c_gsize = int(cols / color_slicing)
    
    bgr_hist = []
    hsv_hist = []
    h_hist = []
    s_hist = []
    v_hist = []
    
    for i in range(0, rows, r_gsize):
        for j in range(0, cols, c_gsize):
            ## make grids which are close to center weighted more
            row_weight = min(abs(i+(r_gsize//2)), abs(rows-(i+(r_gsize//2)))) // (r_gsize//3)
            col_weight = min(abs(j+(c_gsize//2)), abs(cols-(j+(c_gsize//2)))) // (c_gsize//3)
            total_weight = row_weight+col_weight
            ## bgr_colorspace, hsv_colorspace, independent h&s&v histogram
            bgr_hist.append((cv2.calcHist([image[i : i+r_gsize, j : j+c_gsize]], [0, 1, 2], None, bgr_bins, [0, 256, 0, 256, 0, 256]).flatten()) * total_weight)
            hsv_hist.append((cv2.calcHist([hsvimg[i : i+r_gsize, j : j+c_gsize]], [0, 1, 2], None, hsv_bins, [0, 256, 30, 256, 0, 256]).flatten()) * total_weight)
            h_hist.append((cv2.calcHist([h[i : i+r_gsize, j : j+c_gsize]], [0], None, [256], [0, 256]).flatten()) * total_weight)
            s_hist.append((cv2.calcHist([s[i : i+r_gsize, j : j+c_gsize]], [0], None, [256], [0, 256]).flatten()) * total_weight)
            v_hist.append((cv2.calcHist([v[i : i+r_gsize, j : j+c_gsize]], [0], None, [256], [0, 256]).flatten()) * total_weight) 
            
    bgr_array = np.asarray(bgr_hist).flatten()
    hsv_array = np.asarray(hsv_hist).flatten()
    h_array = np.asarray(h_hist).flatten()
    s_array = np.asarray(s_hist).flatten()
    v_array = np.asarray(v_hist).flatten()
    #concate histogram to be color feature
    color_feature = np.concatenate([bgr_array, hsv_array, h_array, s_array, v_array])
    return color_feature

########### edge feature ###########
def hog(pyramid):
    edge_feature = []
    for i in range(len(pyramid)):
        angle_list = []
        angle_array = np.zeros([angel_slicing])
        image = pyramid[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows = np.shape(image)[0]
        cols = np.shape(image)[1]
        ## slice each image in pyramid into grid(#grid is related to the size of image)
        r_gsize = int(rows/(edge_slicing-i))
        c_gsize = int(cols/(edge_slicing-i))
        
        for r in range(0, rows, r_gsize):
            for c in range(0, cols, c_gsize):
                ## conculate the magnitude&angel of each grid, and then find the histogram of angel which is weighted by magnitude
                gx = cv2.Sobel(image[r : r+r_gsize, c : c+c_gsize], cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(image[r : r+r_gsize, c : c+c_gsize], cv2.CV_32F, 0, 1, ksize=1)
                mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                height = np.shape(mag)[0]
                width = np.shape(mag)[1]
                for h in range(height):
                    for w in range(width):
                        angle_array[int(angle[h][w]/(360/angel_slicing))] += mag[h][w]
                for a in range(len(angle_array)):
                    edge_feature.append(angle_array[a])
    return edge_feature
def pyramid(image, scale=2, minSize=(25, 25)):
    ## construst image pyramid
    pyramid = []
    pyramid.append(image)
    while True:
        w = int(image.shape[1] / scale)
        if w< minSize[0]:
            break
        image = cv2.resize(image, (w, w))
        pyramid.append(image)
    return pyramid

########### local feature ###########
def HOG_sift(nfeatures, image):
    ##get sift keypoint&descriptor
    sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    key , des = sift.detectAndCompute(image, None)
    return des


########### evaluate ###########
def Bhattacharyya_distance(vec_a, vec_b):
    BC=np.sum(np.sqrt(vec_a*vec_b))
    return np.log(BC)
def coosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b))
def evaluate(feature, distance_flag):
    AP = np.zeros([700])
    for i in range(700):
        print(i, end='\r')
        score = {}
        for j in range(700):
            if i!=j:
                ## color feature is evaluted by Bhattacharyya_distance, and local&edge feature are evaluted by coosine_similarity
                if distance_flag==1:
                    score[j] = coosine_similarity(feature[i], feature[j])
                elif distance_flag==2:
                    score[j] = Bhattacharyya_distance(feature[i], feature[j])
        ## calculate AP
        num_of_retrive = 0
        num_of_correct = 0
        Precision = 0
        sortlist =  sorted(score, key=score.get, reverse=True)
        for j in sortlist:
            num_of_retrive += 1
            if (i//20)==(j//20): ##代表同一個category
                num_of_correct += 1
                Precision += (num_of_correct/num_of_retrive)
        AP[i] = Precision/19
    return AP
def evaluate_fusion(color_feature, edge_feature, local_feature):
    AP = np.zeros([700])
    for i in range(700):
        print(i, end='\r')
        i_category = i//20
        score_c = {}
        score_e = {}
        score_l = {}
        ## for each image, we calculate score_c & score_e & score_l, which means:
        ## score_c:use color feature
        ## score_e:use color feature + edge feature
        ## score_l:use color feature + local feature
        for j in range(700):
            if i!=j:
                score_c[j] = Bhattacharyya_distance(color_feature[i], color_feature[j])          
                score_e[j] = coosine_similarity(edge_feature[i], edge_feature[j])*3 + Bhattacharyya_distance(color_feature[i], color_feature[j])
                score_l[j] = coosine_similarity(local_feature[i], local_feature[j]) + Bhattacharyya_distance(color_feature[i], color_feature[j])
                   
        ## sum the ranks of image j in sortlist_c&sortlist_e&sortlist_l, consider the j with smallest rank as the image which is most relate to i
        total_rank_array = np.zeros([700])
        total_rank = {}
        sortlist_c =  sorted(score_c, key=score_c.get, reverse=True)
        sortlist_e =  sorted(score_e, key=score_e.get, reverse=True)
        sortlist_l =  sorted(score_l, key=score_l.get, reverse=True)
        for j in range(699):
            total_rank_array[sortlist_c[j]] += j
            total_rank_array[sortlist_e[j]] += j
            total_rank_array[sortlist_l[j]] += j
        for j in range(699):
            total_rank[j] = total_rank_array[j]
        sortlist = sorted(total_rank, key=total_rank.get)
        num_of_retrive = 0
        num_of_correct = 0
        Precision = 0
        for j in sortlist:
            if(i!=j):
                num_of_retrive += 1
                if (i//20)==(j//20): ##代表同一個category
                    num_of_correct += 1
                    Precision += (num_of_correct/num_of_retrive)
        AP[i] = Precision/19
    return AP
def MAP(namelist, AP):
    ## print MAP of each category, show the max&mean
    MAP_dict = {}
    MAP_array = np.zeros([35])
    for i in range(700):
        MAP_array[i//20] += AP[i]
        if i%20==19:
            MAP_array[i//20] /= 20
    for i in range(35):
        MAP_dict[i] = MAP_array[i]
    for i in  sorted(MAP_dict, key=MAP_dict.get, reverse=True):
        print(namelist[i][0], " : ", MAP_array[i])
    print("---------------------------------------")
    print("mean: ", np.mean(MAP_array))
    print("max: ", max(MAP_array))
    return MAP_array

############讀檔############
Folder = sys.argv[1]
filename = os.path.join(Folder, "database")
##namelist 第一個 column 存 category 的資料夾名稱，其餘 20 個 column 為圖片檔名
##namelist 共有 35 個 row
namelist = read_file(filename)
namelist_700 = []
for i in range(len(namelist)):
    category_path = os.path.join(filename, namelist[i][0])
    for j in range(1, len(namelist[i])):
        namelist_700.append(os.path.join(category_path, namelist[i][j]))

############color feature############
color_featurelist = []
for i in range(len(namelist)):
    category_path = os.path.join(filename, namelist[i][0])
    for j in range(1, len(namelist[i])):
        image = cv2.imread(os.path.join(category_path, namelist[i][j]))
        res_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        crop_img = res_img[28:28+200, 28:28+200]
        color_feature = color_hist(crop_img)
        color_featurelist.append(color_feature)
print("color_featuresize :", np.shape(color_featurelist))
color_featurearray = np.asarray(color_featurelist)
np.save('color_feature', color_featurearray)

############edge feature############
edge_featurelist = []
for i in range(len(namelist)):
    print(i, end = '\r')
    category_path = os.path.join(filename, namelist[i][0])
    for j in range(1, len(namelist[i])):
        image = cv2.imread(os.path.join(category_path, namelist[i][j]))
        res_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        crop_img = res_img[28:28+200, 28:28+200]
        image_pyramid = pyramid(crop_img)
        edge_feature = hog(image_pyramid)
        edge_featurelist.append(edge_feature)

print("edge_featuresize :", np.shape(edge_featurelist))
edge_featurearray = np.asarray(edge_featurelist)
np.save('edge_feature', edge_featurearray)

############local feature############
sift_featurelist = [] ## get less sift descriptor for training kmeans
for i in range(len(namelist)):
    category_path = os.path.join(filename, namelist[i][0])
    print(i, end = '\r')
    for j in range(1, len(namelist[i])):
        image = cv2.imread(os.path.join(category_path, namelist[i][j]))
        res_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        sift_feature = HOG_sift(nfeatures_less, res_img)
        for f in sift_feature:
            sift_featurelist.append(f)
print("sift_featuresize :", np.shape(sift_featurelist))

sift_featurearray = np.array(sift_featurelist)
kmeans = KMeans(n_clusters = cluster_nums, random_state = 0).fit(sift_featurearray)

local_featurelist = []
for i in range(len(namelist)):
    category_path = os.path.join(filename, namelist[i][0])
    print(i, end = '\r')
    for j in range(1, len(namelist[i])):
        image = cv2.imread(os.path.join(category_path, namelist[i][j]))
        res_img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        sift_feature2 = HOG_sift(nfeatures_more, res_img) ## get more sift descriptor for testing kmeans
        ##算圖片中的每個 descriptor 屬於哪個 cluster，local_feature 統計每個 cluster 有幾個 descriptor
        des_clusters = kmeans.predict(sift_feature2)
        local_feature = np.zeros([cluster_nums])
        for k in range(len(des_clusters)):
            local_feature[des_clusters[k]] += 1
        local_featurelist.append(local_feature)
print("local_featuresize :", np.shape(local_featurelist))
local_featurearray = np.asarray(local_featurelist)
np.save("local_feature", local_featurearray)

############evaluate color############
print("############ evaluate color ############")
color_feature = np.load('color_feature.npy')
print("color_featuresize :", np.shape(color_feature))
AP_color = evaluate(color_feature, 2)
print("===== Color MAP from high to low =====")
MAP_color = MAP(namelist, AP_color)
print("\n")

############evaluate edge############
print("############ evaluate edge ############")
edge_feature = np.load('edge_feature.npy')
print("edge_featuresize :", np.shape(edge_feature))
AP_edge = evaluate(edge_feature, 1)
print("===== Edge MAP from high to low =====")
MAP_edge = MAP(namelist, AP_edge)
print("\n")

############evaluate local############
print("############ evaluate local ############")
local_feature = np.load('local_feature.npy')
print("local_featuresize :", np.shape(local_feature))
AP_local = evaluate(local_feature, 1)
print("===== Local MAP from high to low =====")
MAP_local = MAP(namelist, AP_local)
print("\n")

############evaluate fusion############
print("############ evaluate fusion ############")
color_feature = np.load('color_feature.npy')
edge_feature = np.load('edge_feature.npy')
local_feature = np.load('local_feature.npy')
AP_fusion = evaluate_fusion(color_feature, edge_feature, local_feature)
print("===== Fusion MAP from high to low =====")
MAP_fusion = MAP(namelist, AP_fusion)
print("\n")