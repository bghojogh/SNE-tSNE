import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle
import matplotlib.pyplot as plt
import glob
from sklearn.metrics.pairwise import pairwise_kernels


class My_SNE:

    def __init__(self, X, y=None, n_components=2, learning_rate=0.1, max_iterations=1000, step_checkpoint=20):
        # X: rows are features and columns are samples
        # labels y is only for plotting the embeddings --> if set None, it does not plot
        self.n_components = n_components
        self.X = X
        self.y = y
        self.n_training_images = self.X.shape[1]
        self.data_dimension = self.X.shape[0]
        self.max_iterations = max_iterations
        self.step_checkpoint = step_checkpoint
        self.learning_rate = learning_rate

    def fit_transform(self, continue_from_which_iteration=None):
        path_to_save = './saved_files/SNE/'
        if continue_from_which_iteration is not None:
            paths_ = glob.glob(path_to_save+'X_transformed/*')
            paths_ = [path_.split("\\")[-1] for path_ in paths_]
            paths_ = [path_.split(".")[0] for path_ in paths_]
            name_of_variable = [path_ for path_ in paths_ if "itr"+str(continue_from_which_iteration) in path_][0]
            X_transformed = self.load_variable(name_of_variable=name_of_variable, path=path_to_save+'X_transformed/')
        else:
            X_transformed = np.random.rand(self.n_components, self.n_training_images)  # --> rand in [0,1)
            # save the information at checkpoints:
            self.save_variable(variable=X_transformed, name_of_variable="X_transformed_initial", path_to_save=path_to_save + 'X_transformed/')
            if self.y is not None:
                self.plot_embedding(X=X_transformed, y=self.y, path_save=path_to_save+"training_plots/", name_of_plot="X_transformed_0_.png")
        print("Calculating p for all pairs...")
        distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        p_matrix = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("---processing for image " + str(sample_index1))
            sigma = 1 / (2 ** 0.5)
            d_squared_of_row = (distance_matrix_originalSpace[sample_index1, :] ** 2) / (2 * (sigma ** 2))
            d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
            minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
            minus_d_squared_of_row = -1 * d_squared_of_row
            denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
            for sample_index2 in range(self.n_training_images):
                if sample_index1 != sample_index2:
                    numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                    p = numerator / denominator
                else:
                    p = 0
                p_matrix[sample_index1, sample_index2] = p
        if continue_from_which_iteration is not None:
            iteration_index = continue_from_which_iteration
        else:
            iteration_index = -1
        cost_iters = np.zeros((self.step_checkpoint, 1))
        update = 0
        while True:
            iteration_index = iteration_index + 1
            #----- update alpha:
            if iteration_index < 250:
                alpha = 0.5   
            else:
                alpha = 0.8  
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=X_transformed)
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                d_squared_of_row = (distance_matrix_embeddedSpace[sample_index1, :] ** 2)
                d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
                minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
                minus_d_squared_of_row = -1 * d_squared_of_row
                denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
                for sample_index2 in range(self.n_training_images):
                    if sample_index1 != sample_index2:
                        numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                        q = numerator / denominator
                    else:
                        q = 0
                    q_matrix[sample_index1, sample_index2] = q
            for sample_index1 in range(self.n_training_images):
                X_i_transformed_previousIteration = X_transformed[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    X_j_transformed_previousIteration = X_transformed[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix[sample_index1, sample_index2]
                    p_ji = p_matrix[sample_index2, sample_index1]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    q_ji = q_matrix[sample_index2, sample_index1]
                    gradient = gradient + (p_ij - q_ij + p_ji - q_ji) * (X_i_transformed_previousIteration - X_j_transformed_previousIteration)
                gradient = gradient * 2
                update = - (self.learning_rate * gradient) + (alpha * update)
                X_i_transformed = X_i_transformed_previousIteration + update
                X_transformed[:, sample_index1] = X_i_transformed.ravel()
            #--- add some jitter:
            if iteration_index < 50:
                for sample_index in range(self.n_training_images):
                    noise = np.random.normal(0, 0.1, self.n_components)
                    X_transformed[:, sample_index] = X_transformed[:, sample_index] + noise
            #--- calculate cost:
            cost = 0
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index2 != sample_index1:
                        p_ij = p_matrix[sample_index1, sample_index2]
                        q_ij = q_matrix[sample_index1, sample_index2]
                        if p_ij != 0 and q_ij != 0:
                            cost = cost + (p_ij * np.log10(p_ij)) - (p_ij * np.log10(q_ij))
            print("---- cost of this iteration: " + str(cost))
            index_to_save = iteration_index % self.step_checkpoint
            cost_iters[index_to_save] = cost
            # save the information at checkpoints:
            if (iteration_index+1) % self.step_checkpoint == 0:
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / self.step_checkpoint))
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_variable(variable=X_transformed, name_of_variable="X_transformed_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'X_transformed/')
                if self.y is not None:
                    self.plot_embedding(X=X_transformed, y=self.y, path_save=path_to_save+"training_plots/", name_of_plot="X_transformed_itr"+str(iteration_index)+"_cp"+str(checkpoint_index)+".png")
            # --- check terminate:
            if self.max_iterations is not None:
                if iteration_index > self.max_iterations:
                    return X_transformed

    def fit_transform_symmetric(self, continue_from_which_iteration=None):
        path_to_save = './saved_files/SNE_symmetric/'
        if continue_from_which_iteration is not None:
            paths_ = glob.glob(path_to_save+'X_transformed/*')
            paths_ = [path_.split("\\")[-1] for path_ in paths_]
            paths_ = [path_.split(".")[0] for path_ in paths_]
            name_of_variable = [path_ for path_ in paths_ if "itr"+str(continue_from_which_iteration) in path_][0]
            X_transformed = self.load_variable(name_of_variable=name_of_variable, path=path_to_save+'X_transformed/')
        else:
            X_transformed = np.random.rand(self.n_components, self.n_training_images)  # --> rand in [0,1)
            # save the information at checkpoints:
            self.save_variable(variable=X_transformed, name_of_variable="X_transformed_initial", path_to_save=path_to_save + 'X_transformed/')
            if self.y is not None:
                self.plot_embedding(X=X_transformed, y=self.y, path_save=path_to_save+"training_plots/", name_of_plot="X_transformed_0_.png")
        print("Calculating p for all pairs...")
        distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        p_matrix = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("---processing for image " + str(sample_index1))
            sigma = 1 / (2 ** 0.5)
            d_squared_of_row = (distance_matrix_originalSpace[sample_index1, :] ** 2) / (2 * (sigma ** 2))
            d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
            minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
            minus_d_squared_of_row = -1 * d_squared_of_row
            denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
            for sample_index2 in range(self.n_training_images):
                if sample_index1 != sample_index2:
                    numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                    p = numerator / denominator
                else:
                    p = 0
                p_matrix[sample_index1, sample_index2] = p
        # make p symmetric:
        p_matrix_symmetric = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            for sample_index2 in range(self.n_training_images):
                p_matrix_symmetric[sample_index1, sample_index2] = (p_matrix[sample_index1, sample_index2] + p_matrix[sample_index2, sample_index1]) / (2 * self.n_training_images)
        if continue_from_which_iteration is not None:
            iteration_index = continue_from_which_iteration
        else:
            iteration_index = -1
        cost_iters = np.zeros((self.step_checkpoint, 1))
        update = 0
        while True:
            iteration_index = iteration_index + 1
            #----- update alpha:
            if iteration_index < 250:
                alpha = 0.5   
            else:
                alpha = 0.8  
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=X_transformed)
            d_squared_of_all = (distance_matrix_embeddedSpace[:, :] ** 2)
            minus_d_squared_of_all = -1 * d_squared_of_all
            denominator = np.sum(np.exp(minus_d_squared_of_all))
            denominator = denominator - np.sum(np.exp(np.diag(minus_d_squared_of_all))) #--> remove the diagonal elements
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                d_squared_of_row = (distance_matrix_embeddedSpace[sample_index1, :] ** 2)
                minus_d_squared_of_row = -1 * d_squared_of_row
                for sample_index2 in range(self.n_training_images):
                    if sample_index1 != sample_index2:
                        numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                        q = numerator / denominator
                    else:
                        q = 0
                    q_matrix[sample_index1, sample_index2] = q
            for sample_index1 in range(self.n_training_images):
                X_i_transformed_previousIteration = X_transformed[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    X_j_transformed_previousIteration = X_transformed[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix_symmetric[sample_index1, sample_index2]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    gradient = gradient + (p_ij - q_ij) * (X_i_transformed_previousIteration - X_j_transformed_previousIteration)
                gradient = gradient * 4
                update = - (self.learning_rate * gradient) + (alpha * update)
                # update = - (self.learning_rate * gradient)
                X_i_transformed = X_i_transformed_previousIteration + update
                X_transformed[:, sample_index1] = X_i_transformed.ravel()
            # #--- add some jitter:
            # if iteration_index < 50:
            #     for sample_index in range(self.n_training_images):
            #         noise = np.random.normal(0, 0.1, self.n_components)
            #         X_transformed[:, sample_index] = X_transformed[:, sample_index] + noise
            #--- calculate cost:
            cost = 0
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index2 != sample_index1:
                        p_ij = p_matrix_symmetric[sample_index1, sample_index2]
                        q_ij = q_matrix[sample_index1, sample_index2]
                        if p_ij != 0 and q_ij != 0:
                            cost = cost + (p_ij * np.log10(p_ij)) - (p_ij * np.log10(q_ij))
            print("---- cost of this iteration: " + str(cost))
            index_to_save = iteration_index % self.step_checkpoint
            cost_iters[index_to_save] = cost
            # save the information at checkpoints:
            if (iteration_index+1) % self.step_checkpoint == 0:
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / self.step_checkpoint))
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_variable(variable=X_transformed, name_of_variable="X_transformed_itr"+str(iteration_index)+"_cp"+str(checkpoint_index), path_to_save=path_to_save+'X_transformed/')
                if self.y is not None:
                    self.plot_embedding(X=X_transformed, y=self.y, path_save=path_to_save+"training_plots/", name_of_plot="X_transformed_itr"+str(iteration_index)+"_cp"+str(checkpoint_index)+".png")
            # --- check terminate:
            if self.max_iterations is not None:
                if iteration_index > self.max_iterations:
                    return X_transformed

    def get_distances_btw_points(self, data_matrix):
        # data_matrix: rows are features and columns are samples
        n_samples = data_matrix.shape[1]
        distance_matrix = KNN(X=data_matrix.T, n_neighbors=n_samples-1, mode='distance', include_self=False, n_jobs=-1)
        distance_matrix = distance_matrix.toarray()
        return distance_matrix

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def plot_embedding(self, X, y, path_save=None, name_of_plot=None):
        # X: column-wise
        color_map = plt.cm.jet  #--> hsv, brg (good for S curve), rgb, jet, gist_ncar (good for one blob), tab10, Set1, rainbow, Spectral #--> https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
        plt.scatter(X[0, :], X[1, :], c=y, cmap=color_map, edgecolors='k')
        classes = [str(i) for i in range(len(np.unique(y)))]
        n_classes = len(classes)
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        if path_save is None:
            plt.show()
        else:
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            plt.savefig(path_save+name_of_plot)
            plt.close()

    def transform_outOfSample(self, X_test, which_training_iteration_to_load, symmetric_method=False):
        # X_test, X_test_transformed: rows are features and columns are samples
        ##### read the training embedding:
        X_transformed = self.read_the_saved_training_embedding(which_training_iteration_to_load, symmetric_method)
        X_transformed = X_transformed.T  #--> make it row-wise
        ##### embedding the out-of-sample:
        kernel_X_X = pairwise_kernels(X=self.X.T, Y=self.X.T, metric="rbf")
        kernel_Xtest_X = pairwise_kernels(X=X_test.T, Y=self.X.T, metric="rbf")
        n_training_samples = self.X.shape[1]
        K = np.zeros((n_training_samples, n_training_samples))
        for sample_index in range(n_training_samples):
            K[sample_index, :] = kernel_X_X[sample_index, :] * (1 / np.sum(kernel_X_X[sample_index, :]))
        n_test_samples = X_test.shape[1]
        K_test = np.zeros((n_test_samples, n_training_samples))
        for test_sample_index in range(n_test_samples):
            K_test[test_sample_index, :] = kernel_Xtest_X[test_sample_index, :] * (1 / np.sum(kernel_Xtest_X[test_sample_index, :]))
        A = np.linalg.pinv(K) @ X_transformed
        X_test_transformed = K_test @ A
        X_test_transformed = X_test_transformed.T  #--> make it column-wise
        return X_test_transformed

    def read_the_saved_training_embedding(self, which_training_iteration_to_load, symmetric_method=False):
        # X_transformed: column-wise
        if not symmetric_method:
            path_to_save = './saved_files/SNE/'
        else:
            path_to_save = './saved_files/SNE_symmetric/'
        paths_ = glob.glob(path_to_save+'X_transformed/*')
        paths_ = [path_.split("\\")[-1] for path_ in paths_]
        paths_ = [path_.split(".")[0] for path_ in paths_]
        name_of_variable = [path_ for path_ in paths_ if "itr"+str(which_training_iteration_to_load) in path_][0]
        X_transformed = self.load_variable(name_of_variable=name_of_variable, path=path_to_save+'X_transformed/')
        return X_transformed