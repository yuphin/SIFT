import os
import cv2
import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix


def best_configuration():
    kmean = 128
    KNN = 16
    step_size = 3
    kmeans = KMeans(kmean, random_state=0)
    ftrs_train = extract_feature_points('train',1,step_size,0,3,0.014,10,0.27)
    ftrs_test = extract_feature_points('validation',1,step_size,0,3,0.014,10,0.27)
    predicted_train = kmeans.fit_predict(ftrs_train[0])
    predicted_test = kmeans.predict(ftrs_test[0])
    vocab_train = create_bof_representation(1, ftrs_train[1], predicted_train, kmean)
    vocab_test = create_bof_representation(1, ftrs_test[1], predicted_test, kmean)
    print(calculate_accuracy(KNN, vocab_test, vocab_train, ftrs_train[2], ftrs_test[2]))
    print('----------')

# Function that extracts descriptors
def extract_feature_points(dir_name, sift_type, step_size=10, nfeatures=0,
                           nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers
                                       , contrastThreshold=contrastThreshold,
                                       edgeThreshold=edgeThreshold, sigma=sigma)
    result = None
    idxs = []
    categories = []
    folder_names = os.listdir(dir_name)
    cnt = 0
    kp1 = []

    for i in range(step_size, 32 - step_size, step_size):
        for j in range(step_size, 32 - step_size, step_size):
            kp1.append(cv2.KeyPoint(i, j, step_size))

    for category in folder_names:
        print(category)

        for file_name in os.listdir(dir_name + '/' + category):

            img = cv2.imread(dir_name + '/' + category + '/' + file_name, 0)
            if sift_type:
                kp, img1_descriptor = sift.compute(img, kp1)
            else:
                img1_kp, img1_descriptor = sift.detectAndCompute(img, None)
                if img1_descriptor is None:
                    kp, img1_descriptor = sift.compute(img, kp1)
            if result is None:
                result = np.array(img1_descriptor)
                idxs.append(img1_descriptor.shape[0])
                categories.append(cnt)
            else:
                result = np.vstack((result, img1_descriptor))
                idxs.append(img1_descriptor.shape[0])
                categories.append(cnt)
        cnt += 1
    return result, idxs, categories,len(idxs)

# Creation of Bag of Features representation using the dictionary
def create_bof_representation(is_dense, idxs, clustered, K):
    ranged_f = np.arange(K)
    # If it's Dense-SIFT, no need to loop over since we have fix amount of keypoints
    if is_dense:
        a = np.reshape(clustered, (idxs[0], len(clustered) // idxs[0]))
        i = np.arange(len(clustered))
        tmp = np.histogram2d(i, a.flatten(), (a.shape[1], ranged_f))
        res = np.apply_along_axis(lambda x: x / (sum(x)), 1, tmp[0])
        return res
    else:
        # Else, we need to loop over each elements to see how much keypoints each image sample have.
        st = 0
        vocab = None

        for idx in idxs:
            sliced = clustered[st:st + idx]
            tmp = np.histogram(sliced, bins=ranged_f)
            normalized = tmp[0] / sum(tmp[0])
            if vocab is None:
                vocab = np.array(normalized)
            else:
                vocab = np.vstack((vocab, normalized))
            st += idx

        return vocab


# For outputting the test_results
def predict_test_data(kmeans,vocab_train,train_labels):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3
                                       , contrastThreshold=0.014,
                                       edgeThreshold=10, sigma=0.27)
    result = None
    step_size = 3
    size = 0
    kp1 = []
    dct = []
    mapping = ['apple','aquarium_fish','beetle','camel','crab','cup',
               'elephant','flatfish','lion','mushroom','orange','pear','road',
               'skyscraper','woman']
    for i in range(step_size, 32 - step_size, step_size):
        for j in range(step_size, 32 - step_size, step_size):
            kp1.append(cv2.KeyPoint(i, j, step_size))
    i =0
    for filename in os.listdir('test'):
        print(i)
        img = cv2.imread('test/'+filename, 0)
        kp, img1_descriptor = sift.compute(img, kp1)
        size = img1_descriptor.shape[0]
        if result is None:
            result = np.array(img1_descriptor)
        else:
            result = np.vstack((result, img1_descriptor))
        dct.append((filename))
        i+=1
    predicted_test = kmeans.predict(result)
    vocab_test = create_bof_representation(1, [size], predicted_test, 128)
    a = []
    np.apply_along_axis(lambda x: a.append(np.argmax(
        np.bincount(np.take(train_labels, np.take(np.argsort(np.linalg.norm(x - vocab_train, axis=1)), np.arange(16)))))),
                        1, vocab_test)
    f = open('test_results.txt','w+')
    for i in range(len(a)):
        f.write(dct[i]+': '+mapping[a[i]]+'\n')
    f.close()




def calculate_accuracy(K, test_data, train_data, train_labels, test_labels):
    a = []
    # The KNN calculation.
    ''' 
    To calculate KNN, we first get the euclidean distance between training data and each element as histogram from
    test data. Then we sort the elements by their arguments using numpy's argsort function. Afterwards we take
    the N from KNN amount of elements. Then we get the element with the most recurrences via bincount and argmax. At 
    the end, we get the category value as an integer between 0 and 14.  
    
    
    '''
    np.apply_along_axis(lambda x: a.append(np.argmax(
        np.bincount(np.take(train_labels, np.take(np.argsort(np.linalg.norm(x - train_data, axis=1)), np.arange(K)))))),
                        1, test_data)
    cm = confusion_matrix(test_labels,a,np.arange(15))
    print(cm)
    dif = (np.count_nonzero(np.array(a) - np.array(test_labels)) / (len(test_labels)))*100

    return 100-dif
# Saving the SIFT features to disk for reuse, heavily used during the experiments to save time
def generate_sift_features(is_dense, step_size=0, nfeatures=0,
                           nOctaveLayers=3,
                           contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    train_vars = \
        extract_feature_points('train', is_dense,
                               step_size,
                               nfeatures, nOctaveLayers,
                               contrastThreshold, edgeThreshold, sigma)
    test_vars = \
        extract_feature_points('validation', is_dense, step_size,
                               nfeatures, nOctaveLayers,
                               contrastThreshold, edgeThreshold, sigma)

    if is_dense:
        train_vars = (train_vars[0].flatten(), train_vars[1], train_vars[2])
        test_vars = (test_vars[0].flatten(), test_vars[1], test_vars[2])
        np.savez('sift/trainvars_step%f_features%f_nOctLayers%f_contrThres%f_edgeThres%f_sigma%f'
                 % (step_size, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma), (train_vars),
                 (test_vars))
    else:
        np.save('sift/trainvars',(train_vars,test_vars))

# Load sift features from a file, used during the experiments to save time
def load_sift_features(is_dense, step_size=0, nfeatures=0,
                       nOctaveLayers=3,
                       contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    if is_dense:
        mfile = np.load('sift/trainvars_step%f_features%f_nOctLayers%f_contrThres%f_edgeThres%f_sigma%f.npz'
                        % (step_size, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma))
        files = mfile.files
        r1 = len(mfile[files[0]][0]) // 128
        r2 = len(mfile[files[1]][0]) // 128
        ar1 = np.reshape(mfile[files[0]][0],(r1,128))
        ar2 = np.reshape(mfile[files[1]][0],(r2,128))
        return (ar1, mfile[files[0]][1],mfile[files[0]][2],ar2,mfile[files[1]][1],mfile[files[1]][2])
    else:
        mfile = np.load('sift/trainvars.npy')
        return  mfile[0][0],mfile[0][1], mfile[0][2],mfile[1][0] , mfile[1][1], mfile[1][2]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'best-configuration':
        best_configuration()
        exit(0)
    K_means = [128]
    K_neighbors = [16]
    step_size = [3]
    batch_size = 35000
    is_dense = 1
    nfeatures = 0
    nOctaveLayers = 3
    contrastThreshold = 0.014
    edgeThreshold = 10
    sigma = 0.27
    # Main loop for the experiments(except the 'test' and 'best-configuration')
    # Used during the experiments
    for kmean in K_means:
        for kneigh in K_neighbors:
            for step in step_size:
                print(kmean, kneigh, batch_size, step)
                ftrs = load_sift_features(is_dense,step,
                                          nfeatures, nOctaveLayers,
                                          contrastThreshold, edgeThreshold, sigma)
                kmeans = KMeans(kmean, random_state=0)
                predicted_train = kmeans.fit_predict(ftrs[0])
                predicted_test = kmeans.predict(ftrs[3])
                vocab_train = create_bof_representation(is_dense, ftrs[1], predicted_train, kmean)
                if len(sys.argv) > 1 and sys.argv[1] == 'test':
                    predict_test_data(kmeans, vocab_train, ftrs[2])
                    exit(0)
                vocab_test = create_bof_representation(is_dense, ftrs[4], predicted_test, kmean)

                print(calculate_accuracy(kneigh, vocab_test, vocab_train, ftrs[2], ftrs[5]))
                print('----------')
