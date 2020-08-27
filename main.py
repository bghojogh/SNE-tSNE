from my_SNE import My_SNE
from my_tSNE import My_tSNE
from my_tSNE_general import My_tSNE_general
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from skimage.transform import resize


def main():
    #---- settings:
    dataset = "MNIST"  #--> MNIST, ORL_glasses
    method = "tSNE_general_degrees" #--> SNE, SNE_symmetric, tSNE, tSNE_general_degrees
    embed_training_data_again = False
    embed_test_data = True
    embed_again = True
    color_map = plt.cm.jet  #--> hsv, brg (good for S curve), rgb, jet, gist_ncar (good for one blob), tab10, Set1, rainbow, Spectral #--> https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html

    #---- dataset:
    X_train, y_train, X_test, y_test, class_names = read_dataset(dataset=dataset)

    #---- embedding:
    if embed_again:
        X_train_embedded = None
        X_test_embedded = None
        if method == "SNE":
            model_ = My_SNE(X=X_train, y=y_train, n_components=2, learning_rate=0.1, max_iterations=1000, step_checkpoint=5)
            if embed_training_data_again:
                X_train_embedded = model_.fit_transform(continue_from_which_iteration=None)
            which_training_iteration_to_load = 159
            if embed_test_data:
                X_test_embedded = model_.transform_outOfSample(X_test=X_test, which_training_iteration_to_load=which_training_iteration_to_load, symmetric_method=False)
            X_train_embedded = model_.read_the_saved_training_embedding(which_training_iteration_to_load, symmetric_method=False)
        elif method == "SNE_symmetric":
            model_ = My_SNE(X=X_train, y=y_train, n_components=2, learning_rate=100, max_iterations=1000, step_checkpoint=5)
            if embed_training_data_again:
                X_train_embedded = model_.fit_transform_symmetric(continue_from_which_iteration=None)
            which_training_iteration_to_load = 129
            if embed_test_data:
                X_test_embedded = model_.transform_outOfSample(X_test=X_test, which_training_iteration_to_load=which_training_iteration_to_load, symmetric_method=True)
            X_train_embedded = model_.read_the_saved_training_embedding(which_training_iteration_to_load, symmetric_method=True)
        elif method == "tSNE":
            model_ = My_tSNE(X=X_train, y=y_train, n_components=2, learning_rate=100, max_iterations=1000, step_checkpoint=5, early_exaggeration=True)
            if embed_training_data_again:
                X_train_embedded = model_.fit_transform(continue_from_which_iteration=None)
            which_training_iteration_to_load = 99
            if embed_test_data:
                X_test_embedded = model_.transform_outOfSample(X_test=X_test, which_training_iteration_to_load=which_training_iteration_to_load)
            X_train_embedded = model_.read_the_saved_training_embedding(which_training_iteration_to_load)
        elif method == "tSNE_general_degrees":
            model_ = My_tSNE_general(X=X_train, y=y_train, n_components=2, learning_rate=100, learning_rate_forDegree=0.1, max_iterations=1000, step_checkpoint=5, early_exaggeration=True)
            if embed_training_data_again:
                X_train_embedded = model_.fit_transform(continue_from_which_iteration=None)
            which_training_iteration_to_load = 99
            if embed_test_data:
                X_test_embedded = model_.transform_outOfSample(X_test=X_test, which_training_iteration_to_load=which_training_iteration_to_load)
            X_train_embedded = model_.read_the_saved_training_embedding(which_training_iteration_to_load)
        #---- save the embeddings:
        # if X_train_embedded is not None:
        #     save_variable(variable=X_train_embedded, name_of_variable="X_train_embedded", path_to_save='./saved_files/'+dataset+"/"+method+"/")
        #     save_variable(variable=y_train, name_of_variable="y_train", path_to_save='./saved_files/'+dataset+"/"+method+"/")
        if X_test_embedded is not None:
            save_variable(variable=X_test_embedded, name_of_variable="X_test_embedded", path_to_save='./saved_files/'+dataset+"/"+method+"/")
            save_variable(variable=y_test, name_of_variable="y_test", path_to_save='./saved_files/'+dataset+"/"+method+"/")
    else:
        # X_train_embedded = load_variable(name_of_variable="X_train_embedded", path='./saved_files/'+dataset+"/"+method+"/")
        # y_train = load_variable(name_of_variable="y_train", path='./saved_files/'+dataset+"/"+method+"/")
        if os.path.isfile('./saved_files/'+dataset+"/"+method+"/X_test_embedded.pckl"): #--> if the test embedding file exists
            X_test_embedded = load_variable(name_of_variable="X_test_embedded", path='./saved_files/'+dataset+"/"+method+"/")
            y_test = load_variable(name_of_variable="y_test", path='./saved_files/'+dataset+"/"+method+"/")
        else:
            X_test_embedded = None

    #---- plot training embedding:
    if X_train_embedded is not None:
        plt.scatter(X_train_embedded[0, :], X_train_embedded[1, :], c=y_train, cmap=color_map, edgecolors='k')
        classes = class_names
        n_classes = len(classes)
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        plt.show()
    #---- plot test embedding:
    if X_test_embedded is not None:
        plt.scatter(X_train_embedded[0, :], X_train_embedded[1, :], c=y_train, cmap=color_map, alpha=0.07)
        plt.scatter(X_test_embedded[0, :], X_test_embedded[1, :], c=y_test, cmap=color_map, edgecolors='k')
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        plt.show()

def read_dataset(dataset):
    # output data are column-wise (rows are features)
    if dataset == "MNIST":
        subset_of_MNIST = True
        pick_subset_of_MNIST_again = True
        # MNIST_subset_cardinality_training = 200
        MNIST_subset_cardinality_training = 50
        # MNIST_subset_cardinality_testing = 10
        MNIST_subset_cardinality_testing = 50
        path_dataset = "./datasets/MNIST/"
        file = open(path_dataset+'X_train.pckl','rb')
        X_train = pickle.load(file); file.close()
        file = open(path_dataset+'y_train.pckl','rb')
        y_train = pickle.load(file); file.close()
        file = open(path_dataset+'X_test.pckl','rb')
        X_test = pickle.load(file); file.close()
        file = open(path_dataset+'y_test.pckl','rb')
        y_test = pickle.load(file); file.close()
        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                dimension_of_data = 28 * 28
                X_train_picked = np.empty((0, dimension_of_data))
                y_train_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_train[y_train == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_training, :]
                    X_train_picked = np.vstack((X_train_picked, X_class_picked))
                    y_class = y_train[y_train == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_training].reshape((-1, 1))
                    y_train_picked = np.vstack((y_train_picked, y_class_picked))
                y_train_picked = y_train_picked.ravel()
                X_test_picked = np.empty((0, dimension_of_data))
                y_test_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_test[y_test == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_testing, :]
                    X_test_picked = np.vstack((X_test_picked, X_class_picked))
                    y_class = y_test[y_test == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_testing].reshape((-1, 1))
                    y_test_picked = np.vstack((y_test_picked, y_class_picked))
                y_test_picked = y_test_picked.ravel()
                # X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                # X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                # y_train_picked = y_train[0:MNIST_subset_cardinality_training]
                # y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset)
            else:
                file = open(path_dataset+'X_train_picked.pckl','rb')
                X_train_picked = pickle.load(file); file.close()
                file = open(path_dataset+'X_test_picked.pckl','rb')
                X_test_picked = pickle.load(file); file.close()
                file = open(path_dataset+'y_train_picked.pckl','rb')
                y_train_picked = pickle.load(file); file.close()
                file = open(path_dataset+'y_test_picked.pckl','rb')
                y_test_picked = pickle.load(file); file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        X_train = X_train.T / 255
        X_test = X_test.T / 255
        class_names = [str(i) for i in range(10)]
    elif dataset == "ORL_glasses":
        path_dataset = "./datasets/ORL_glasses/"
        n_samples = 400
        scale = 0.5
        image_height = int(112 * scale)
        image_width = int(92 * scale)
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        image_index = -1
        for class_index in range(2):
            for filename in os.listdir(path_dataset + "class" + str(class_index + 1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index + 1) + "/" + filename,
                                 image_height=image_height, image_width=image_width, do_resize=False, scale=scale)
                data[:, image_index] = img.ravel()
                labels[:, image_index] = class_index
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- normalize (standardation):
        X_notNormalized = data
        # data = data / 255
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
        X_train = data
        y_train = labels.ravel()
        X_test = None
        y_test = None
        class_names = ["Without glasses", "With glasses"]
    return X_train, y_train, X_test, y_test, class_names

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def load_image(address_image, image_height, image_width, do_resize=False, scale=1):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    if do_resize:
        size = int(image_height * scale), int(image_width * scale)
        # img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = resize(img_arr, (int(img_arr.shape[0]*scale), int(img_arr.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
    return img_arr

if __name__ == "__main__":
    main()
