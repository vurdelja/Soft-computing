import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn import datasets
import sys

#ucitavanje iz foldera
#folder = sys.argv[1]
folder = "C:\\Users\\Katarina\\Desktop\\softk2\\data1\\"
mae = []

#funkcije za ucitavanje slike
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
def display_image(image):
    plt.imshow(image, 'gray')
    
train_dir = folder + 'pictures'

# razdvajanje na pozitivne i negativne
pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)

# HOG
pos_features = []
neg_features = []
labels = []

nbins = 9 # broj binova
cell_size = (8, 8) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku


def get_hog():
    # Racunanje HOG deskriptora za slike iz MNIST skupa podataka
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
    return hog

hog = get_hog()

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

# Podela trening skupa na trening i validacioni
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Obucavanje i validacija SVM klasifikatora
clf_svm = SVC(kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)


def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]


# # PROCESS IMAGE - da menjam parametre, svaki objekat samo jednom da registrujem, a treba poetncijalno vise kola da registruje,
# #vise kola da registruje i za ista kola da smatra da su ista. Radi samo za jedna kola,
# # izbacim slucajeve kad se ponavljaju kola
def process_image(images, step_size, window_size, threshold):
    all_best_scores = []
    all_best_windows = []

    for image in images:
        best_score = 0
        best_window = None
        found_objects = set()

        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                this_window = (y, x)
                window = image[y:y+window_size[1], x:x+window_size[0]]

                if window.shape == (window_size[1], window_size[0]):
                    # Provera da li smo već pronašli objekat na ovom mestu
                    if this_window in found_objects:
                        continue

                    score = classify_window(window)

                    if score > best_score:
                        best_score = score
                        best_window = this_window

        if best_window is not None:
            all_best_scores.append(best_score)
            all_best_windows.append(best_window)

            # Dodajemo pronađeni objekat u set pronađenih objekata
            found_objects.add(best_window)

    return all_best_scores, all_best_windows


def detect_line(img):
    # detekcija koordinata linije koristeci Hough transformaciju
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Definisanje opsega crvene boje u HSV prostoru
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    # Maskiranje crvene boje
    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Spajanje slika crvene maske i originalne slike
    red_lines_img = cv2.bitwise_and(img, img, mask=red_mask)

    # Pretvaranje u sivu sliku
    gray_img = cv2.cvtColor(red_lines_img, cv2.COLOR_BGR2GRAY)
    
    edges_img = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    plt.imshow(edges_img, "gray")
    
    # minimalna duzina linije
    min_line_length = 200
    
    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    
    print("Detektovane linije [[x1 y1 x2 y2]]: \n", lines)
    
    x1 = lines[0][0][0]
    y1 = 480 - lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = 480 - lines[0][0][3]
    
    return (x1, y1, x2, y2)

def get_line_params(line_coords):
    k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
    n = k * (float(-line_coords[0])) + float(line_coords[1])
    return k, n


def detect_cross(x, y, k, n):
    # detekcija da li je broj presao preko linije
    yy = k*x + n
    
    return -0.5 <= (yy - y) <= 1

# def detect_cross(x, y, k, n, line_coords):
#     crossed_objects = 0


#     x1, y1, x2, y2 = line_coords 

#     k, n = get_line_params(line_coords)
#     print("\nDetektovani parametri jednačine prave:")
#     print("k=", k)
#     print("n=", n)

#     # Računanje y koordinate na liniji (yy) za dati x
#     yy = k * x + n

#     # Provera da li je objekat prešao preko linije
#     crossed_objects = np.sum((-0.5 <= (yy - y)) & (yy - y <= 1))

#     return crossed_objects




# # PROCESS VIDEO, vise nema k i n i framenum=1 uvek se sve radi posto se linija pomera
# # treshold, klasifikatore, donji doe koda ne treba, Pozivam porces image i detectline
# # pravim svoju funkciju za oznaku kad se preklope ta dva
def process_video(video_path):
    sum_of_nums = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)
        
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break
        
        if frame_num == 3:
            line_coords = detect_line(frame)
            
            k, n = get_line_params(line_coords)
            print("\nDetektovani parametri jednačine prave:")
            print("k=", k)
            print("n=", n)
            
            # # izdvajanje krajnjih x koordinata linije
            # line_left_x = line_coords[0]
            # line_right_x = line_coords[2]
         
        
            # # svaki pravougaonik je opisan sa x,y,w,h
            # for rectangle in rectangles:
            #     x, y, w, h = rectangle
            #     if h > 10: 
            #         # isecanje cifre i skaliranje na 28x28
            #         roi = frame_numbers[y:y+h, x:x+w]
            #         roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_NEAREST)
                
            #     # hog
            #     number_features = hog.compute(roi).reshape(1, -1)
            #     predicted_number = classifier.predict(number_features)[0]
                
            #     # centar cifre
            #     center_x = x + w / 2
            #     center_y = 480 - (y + h / 2)
                
            #     if (line_left_x <= center_x <= line_right_x) and (detect_cross(center_x, center_y, k, n)):
            #         sum_of_nums += predicted_number

            window_size = (400, 200)
            step_size = 100
            
                
            result = process_image(frame, step_size, window_size, 0.6)
                
            crossed_objects = detect_cross(x, y, k, n)
            sum_of_nums += crossed_objects
                
        else:
            continue
            
    cap.release()
    return sum_of_nums        
            



suma = process_video(folder + "videos\\segment_1.mp4")

print("Izračunata suma: ", suma)



