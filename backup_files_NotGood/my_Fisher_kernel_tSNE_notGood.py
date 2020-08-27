import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle
import matplotlib.pyplot as plt
import glob


class My_Fisher_kernel_tSNE:

    def __init__(self, X, y, n_components=2, learning_rate=0.1, max_iterations=1000, 
                step_checkpoint=20, early_exaggeration=True):
        # X: rows are features and columns are samples
        self.n_components = n_components
        self.X = X
        self.y = y
        self.n_training_images = self.X.shape[1]
        self.data_dimension = self.X.shape[0]
        self.max_iterations = max_iterations
        self.step_checkpoint = step_checkpoint
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration

    def fit_transform(self, continue_from_which_iteration=None):
        path_to_save = './saved_files/Fisher_kernel_tSNE/'
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
        # distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        distance_matrix_originalSpace = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("///processing for image " + str(sample_index1))
            for sample_index2 in range(self.n_training_images):
                if sample_index1 != sample_index2:
                    distance_matrix_originalSpace[sample_index1, sample_index2] = self.T_point_approximation(sample_index1, sample_index2, T=10)
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
            #----- early exagaration:
            if self.early_exaggeration:
                if iteration_index < 10:
                    p_matrix_symmetric_used = p_matrix_symmetric * 4   
                else:
                    p_matrix_symmetric_used = p_matrix_symmetric
            else:
                p_matrix_symmetric_used = p_matrix_symmetric
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=X_transformed)
            d_squared_of_all = (distance_matrix_embeddedSpace[:, :] ** 2)
            d_squared_of_all_plus1 = d_squared_of_all + 1
            d_squared_of_all_plus1_inverse = np.reciprocal(d_squared_of_all_plus1) #--> element-wise inverse
            denominator = np.sum(d_squared_of_all_plus1_inverse)
            denominator = denominator - np.sum(np.diag(d_squared_of_all_plus1_inverse)) #--> remove the diagonal elements
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index1 != sample_index2:
                        numerator = d_squared_of_all_plus1_inverse[sample_index1, sample_index2]
                        q = numerator / denominator
                    else:
                        q = 0
                    q_matrix[sample_index1, sample_index2] = q
            for sample_index1 in range(self.n_training_images):
                X_i_transformed_previousIteration = X_transformed[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    X_j_transformed_previousIteration = X_transformed[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix_symmetric_used[sample_index1, sample_index2]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    gradient = gradient + (p_ij - q_ij) * (X_i_transformed_previousIteration - X_j_transformed_previousIteration) * (1 / (1 + (np.linalg.norm(X_i_transformed_previousIteration - X_j_transformed_previousIteration)**2)))
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
                        p_ij = p_matrix_symmetric_used[sample_index1, sample_index2]
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

    def T_point_approximation(self, sample_index1, sample_index2, T=10):
        x1 = self.X[:, sample_index1].reshape((-1, 1))
        x2 = self.X[:, sample_index2].reshape((-1, 1))
        d_T = 0
        for t in range(1, T+1):
            point1 = x1 + ((t-1)/T)*(x2-x1)
            point2 = x1 + (t/T)*(x2-x1)
            J = self.calculate_J(x=point1, label_of_x=self.y[sample_index1])
            d1 = ((point1 - point2).T @ J @ (point1 - point2))**0.5
            d_T += d1
        return d_T

    def calculate_J(self, x, label_of_x):
        sigma = 1 / (2 ** 0.5)
        distance_vector_originalSpace = np.array([np.linalg.norm(x - self.X[:, i].reshape(-1, 1)) for i in range(self.n_training_images)])
        d_squared_of_row = (distance_vector_originalSpace ** 2) / (2 * (sigma ** 2))
        expectation_supervised = 0
        expectation_unsupervised = 0
        for sample_index_i in range(self.n_training_images):
            d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index_i)   # remove the sample_index1-th column of minus_d_of_row
            minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
            minus_d_squared_of_row = -1 * d_squared_of_row
            denominator_unsupervised = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
            numerator_unsupervised = np.exp(minus_d_squared_of_row[sample_index_i])  
            zi_unsupervised = numerator_unsupervised / denominator_unsupervised
            mask_classes = [1 if self.y[i] == label_of_x else 0 for i in range(self.n_training_images)]
            denominator_supervised = np.sum(np.multiply(denominator_unsupervised, mask_classes))
            numerator_supervised = np.exp(minus_d_squared_of_row[sample_index_i]) * mask_classes[sample_index_i]
            zi_supervised = numerator_supervised / denominator_supervised
            expectation_supervised += zi_supervised * self.X[:, sample_index_i].reshape((-1, 1))
            expectation_unsupervised += zi_unsupervised * self.X[:, sample_index_i].reshape((-1, 1))
            b = expectation_supervised - expectation_unsupervised
            p = denominator_supervised / denominator_unsupervised
            expectation_b_bTranspose = p * (b @ b.T)
            J = expectation_b_bTranspose / (sigma ** 4)
        return J