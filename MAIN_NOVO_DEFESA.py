from GENERATE_REPRESENTATIONS_DEFESA import Representations
import cv2
import numpy as np
import os
import dlib
from imutils import face_utils
import math

size_input_data = [96, 96, 1]
#unlabeled_path = './kyoto/'
#unlabeled_path = '/dataset/kyoto/'
unlabeled_path = '/dataset/kyoto/'
#labeled_path = './CKold/'
labeled_path = '/dataset/CKold/'
#labeled_path_jaffe = './jaffe/'
labeled_path_jaffe = '/dataset/jaffe/'
landmarks_predictor_model = './shape_predictor_68_face_landmarks.dat'
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_predictor_model)

def load_unlabeled_database(dir_path):
    size_input_data = [96, 96, 1]
    cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    img_list = os.listdir(dir_path)
    img_data_list = []

    if '.' in img_list[0]:
        for img in img_list:
            input_img = cv2.imread(dir_path + img, cv2.IMREAD_GRAYSCALE)
            input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
            input_img = np.reshape(input_img, (size_input_data[0], size_input_data[1], size_input_data[2]))
            img_data_list.append(input_img)
    else:
        for dataset in img_list:
            imgs = os.listdir(dir_path + '/' + dataset)
            for img in imgs:
                input_img = cv2.imread(dir_path + '/' + dataset + '/' + img, cv2.IMREAD_GRAYSCALE)

                rects = cascade.detectMultiScale(input_img, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (50, 50))

                #if len(rects) > 0:
                #    facerect = rects[0]
                #    input_img = input_img[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]

                input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
                input_img = np.reshape(input_img, (size_input_data[0], size_input_data[1], size_input_data[2]))
                img_data_list.append(input_img)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32') / 255.0
    n_train = int(img_data.shape[0] * 0.7)
    trainX, testX = img_data[:n_train, :], img_data[n_train:, :]

    return trainX, testX

def load_labeled_database(labeled_path):
    image_dir = "cohn-kanade-images"
    label_dir = "Emotion"

    features = []
    labels = np.zeros((327, 1))
    indiv_nomes = []
    counter = 0
    # Maybe sort them
    for participant in os.listdir(os.path.join(labeled_path, image_dir)):
        for sequence in os.listdir(os.path.join(labeled_path, image_dir, participant)):
            if sequence != ".DS_Store":
                    image_files = sorted(os.listdir(os.path.join(labeled_path, image_dir, participant, sequence)))
                    image_file = image_files[-1]
                    input_img = cv2.imread(os.path.join(labeled_path, image_dir, participant, sequence, image_file))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                    rects = cascade.detectMultiScale(input_img, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (150, 150))
                    if len(rects) > 0:
                        facerect = rects[0]
                        input_img = input_img[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]
                        input_img = cv2.resize(input_img, (size_input_data[0], size_input_data[1]))
                        # input_img = np.reshape(input_img, (size_to_resize[0], size_to_resize[1], size_to_resize[2]))
                    features.append(input_img)
                    indiv_nomes.append(participant)
                    label_file = open(
                        os.path.join(labeled_path, label_dir, participant, sequence, image_file[:-4] + "_emotion.txt"))
                    labels[counter] = eval(label_file.read())
                    label_file.close()
                    counter += 1

    print("individuos:",counter)
    img_data = np.array(features)
    img_data_preprocessing = preprocessing(img_data)

    return img_data_preprocessing, labels

def load_labeled_database_jaffe(labeled_path):
    expres_code = ['NE', 'HA', 'AN', 'DI', 'FE', 'SA', 'SU']

    data_dir_list = os.listdir(labeled_path_jaffe)
    counter = 0
    features = []
    labels = np.zeros((213, 1))
    img_names = []

    for dataset in data_dir_list:
        img_list = os.listdir(labeled_path_jaffe + '/' + dataset)
        for img in img_list:
            # imarray = cv2.imread(jaffe_dir + '/' + dataset + '/' + img, cv2.IMREAD_GRAYSCALE)
            imarray = cv2.imread(labeled_path_jaffe + '/' + dataset + '/' + img)
            imarray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)

            rects = cascade.detectMultiScale(imarray, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (150, 150))
            if len(rects) > 0:
                facerect = rects[0]
                imarray = imarray[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]

            imarray = cv2.resize(imarray, (size_input_data[0], size_input_data[1]))
            features.append(imarray)
            label = img[3:5]  # each name of image have 2 char for label from index 3-5
            labels[counter] = expres_code.index(label)
            names = img[0:2]
            img_names.append(names)
            counter += 1
    img_data = np.array(features)
    img_data_preprocessing = preprocessing(img_data)
    return img_data_preprocessing, labels

def preprocessing(input_images):
    normalized_feature_vector_array = []
    for gray in input_images:
        left_eye, rigth_eye = detect_eyes(gray)

        angle = angle_line_x_axis(left_eye, rigth_eye)
        rotated_img = rotateImage(gray, angle)

        # line length
        D = cv2.norm(np.array(left_eye) - np.array(rigth_eye))

        # center of the line
        D_point = [(left_eye[0] + rigth_eye[0]) / 2, (left_eye[1] + rigth_eye[1]) / 2]

        # Face ROI
        x_point = int(D_point[0] - (0.9 * D))
        y_point = int(D_point[1] - (0.6 * D))
        width_point = int(1.8 * D)
        height_point = int(2.2 * D)
        r = [x_point, y_point, width_point, height_point]
        face_roi = rotated_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to (96, 128)
        face_roi = cv2.resize(face_roi, (96, 96))
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = np.reshape(face_roi, (96, 96, 1))

        # Pass through encoder and resize
        #feature_vector = featuremodel_AECNN.predict(face_roi)

        # Reshape feature vector
        shape = face_roi.shape
        aux = 1
        for i in shape:
            aux *= i
        feature_vector = face_roi.reshape(aux)
        normalized_feature_vector_array.append(face_roi)
    normalized_feature_vector_array = np.array(normalized_feature_vector_array)
    return normalized_feature_vector_array

def detect_eyes(gray):
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        pts_right = shape[36: 42]  # right eye landmarks
        pts_left = shape[42: 48]  # left eye landmarks

        hull_right = cv2.convexHull(pts_right)
        M_right = cv2.moments(hull_right)
        # calculate x,y coordinate of center
        cX_right = int(M_right["m10"] / M_right["m00"])
        cY_right = int(M_right["m01"] / M_right["m00"])
        right_eye_center = (cX_right, cY_right)

        hull_left = cv2.convexHull(pts_left)
        M_left = cv2.moments(hull_left)
        # calculate x,y coordinate of center
        cX_left = int(M_left["m10"] / M_left["m00"])
        cY_left = int(M_left["m01"] / M_left["m00"])
        left_eye_center = (cX_left, cY_left)

    return left_eye_center, right_eye_center

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def angle_line_x_axis(point1, point2):
    angle_r = math.atan2(point1[1] - point2[1], point1[0] - point2[0]);
    angle_degree = angle_r * 180 / math.pi;
    return angle_degree

if __name__ == '__main__':
    # carregar base nao rotulada
    trainx, testx = load_unlabeled_database(unlabeled_path)

    #carregar base rotulada
    X_target_CK, Y_target = load_labeled_database(labeled_path)
    X_target_JAFFE, Y_target = load_labeled_database_jaffe(labeled_path_jaffe)

    

    #n_rep = [5, 10, 15, 20, 30, 40, 50, 70, 100, 150]
    n_rep = [5]

    # matriz = np.array([
    #     [True, False, False],  # S
    #     [False, False, True], # L
    #     [True, False, True],  # SL
    #     [True, True, False],  # SA
    #     [True, True, True],  # SLA
    #     [False, True, True]  # LA
    # ])
    # techs = ['S', 'L', 'SL', 'SA', 'SLA', 'LA']
    matriz = np.array([
        [False, False, True], # L
        #[False, True, True], # LA
        #[False, True, False], # A
    ])
    #techs = ['L', 'LA', 'A']
    techs = ['L']


    for i in range(len(n_rep)):
        for j in range(matriz.shape[0]):
            teste = Representations(trainx, testx, X_target_CK, X_target_JAFFE)
            teste.Generate_all(number_of_repr=n_rep[i], seeds_rep=matriz[j, 0], arch_rep=matriz[j, 1], hidden_rep=matriz[j, 2], epochs=50, batch_size=16, techs = techs[j])


