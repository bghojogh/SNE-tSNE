import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle
import matplotlib.pyplot as plt


class My_SNE:

    def __init__(self, X, y=None, n_components=2):
        # X: rows are features and columns are samples
        # labels y is only for plotting the embeddings --> if set None, it does not plot
        self.n_components = n_components
        self.X = X
        self.y = y
        self.n_training_images = self.X.shape[1]
        self.data_dimension = self.X.shape[0]

    def SNE_embed_NOT_EFFICIENT(self, max_iterations=1000, step_checkpoint=20, calculate_again=True):
        if calculate_again == False:
            path_to_save = './output/SNE/'
            Y = self.load_variable(name_of_variable="Y_13", path=path_to_save + 'Y/')
            return Y
        Y = np.random.rand(self.n_components, self.n_training_images)  # --> rand in [0,1)
        # save the information at checkpoints:
        path_to_save = './output/SNE/'
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save + 'Y/')
        print("Calculating p for all pairs...")
        distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        p_matrix = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("---processing for image " + str(sample_index1))
            for sample_index2 in range(self.n_training_images):
                if sample_index1 != sample_index2:
                    p = self.SNE_get_p_NOT_EFFICIENT(i=sample_index1, j=sample_index2, distance_matrix=distance_matrix_originalSpace)
                else:
                    p = 0
                p_matrix[sample_index1, sample_index2] = p
        eta = 0.5
        alpha = 0.5
        iteration_index = -1
        cost_iters = np.zeros((step_checkpoint, 1))
        while True:
            iteration_index = iteration_index + 1
            if iteration_index == 0:
                Y_two_previous_iterations = Y
            else:
                Y_two_previous_iterations = Y_previous_iteration
            Y_previous_iteration = Y
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=Y)
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index1 != sample_index2:
                        q = self.SNE_get_q_NOT_EFFICIENT(i=sample_index1, j=sample_index2, distance_matrix=distance_matrix_embeddedSpace)
                    else:
                        q = 0
                    q_matrix[sample_index1, sample_index2] = q
            for sample_index1 in range(self.n_training_images):
                y_i_previousIteration = Y_previous_iteration[:, sample_index1].reshape((-1, 1))
                y_i_twoPreviousIterations = Y_two_previous_iterations[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    y_j_previousIteration = Y_previous_iteration[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix[sample_index1, sample_index2]
                    p_ji = p_matrix[sample_index2, sample_index1]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    q_ji = q_matrix[sample_index2, sample_index1]
                    gradient = gradient + (p_ij - q_ij + p_ji - q_ji) * (y_i_previousIteration - y_j_previousIteration)
                gradient = gradient * 2
                y_i = y_i_previousIteration - (eta * gradient) + (alpha * (y_i_previousIteration - y_i_twoPreviousIterations))
                Y[:, sample_index1] = y_i.ravel()
            #--- add some jitter:
            if iteration_index < 50:
                for sample_index in range(self.n_training_images):
                    noise = np.random.normal(0, 0.1, self.n_components)
                    Y[:, sample_index] = Y[:, sample_index] + noise
            #--- calculate cost:
            cost = 0
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index2 != sample_index1:
                        p_ij = p_matrix[sample_index1, sample_index2]
                        q_ij = q_matrix[sample_index1, sample_index2]
                        if (p_ij / q_ij) != 0:
                            cost = cost + (p_ij * np.log10(p_ij / q_ij))
            print("---- cost of this iteration: " + str(cost))
            index_to_save = iteration_index % step_checkpoint
            cost_iters[index_to_save] = cost
            # save the information at checkpoints:
            if (iteration_index+1) % step_checkpoint == 0:
                path_to_save = './output/SNE/'
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_variable(variable=Y, name_of_variable="Y_"+str(checkpoint_index), path_to_save=path_to_save+'Y/')
            # --- check terminate:
            if iteration_index > max_iterations:
                return Y

    def get_distances_btw_points(self, data_matrix):
        # data_matrix: rows are features and columns are samples
        n_samples = data_matrix.shape[1]
        distance_matrix = KNN(X=data_matrix.T, n_neighbors=n_samples-1, mode='distance', include_self=False, n_jobs=-1)
        distance_matrix = distance_matrix.toarray()
        return distance_matrix

    def SNE_get_p_NOT_EFFICIENT(self, i, j, distance_matrix, sigma=1/(2**0.5)):
        denominator = 0
        for sample_index2 in range(self.n_training_images):
            if sample_index2 != i:
                distance = distance_matrix[i, sample_index2]
                d_squared = (distance ** 2) / (2 * (sigma**2))
                denominator = denominator + np.exp(-1 * d_squared)
        distance = distance_matrix[i, j]
        d_squared = (distance ** 2) / (2 * (sigma ** 2))
        p = np.exp(-1 * d_squared) / denominator
        return p

    def SNE_get_q_NOT_EFFICIENT(self, i, j, distance_matrix):
        denominator = 0
        for sample_index2 in range(self.n_training_images):
            if sample_index2 != i:
                distance = distance_matrix[i, sample_index2]
                z_squared = (distance ** 2)
                denominator = denominator + np.exp(-1 * z_squared)
        distance = distance_matrix[i, j]
        z_squared = (distance ** 2)
        q = np.exp(-1 * z_squared) / denominator
        return q

    def SNE_get_p(self, i, j, distance_matrix, sigma=1/(2**0.5)):
        d_squared_of_row = (distance_matrix[i, :] ** 2) / (2 * (sigma ** 2))
        minus_d_squared_of_row = -1 * d_squared_of_row
        denominator = np.sum(np.exp(minus_d_squared_of_row)) - np.exp(minus_d_squared_of_row[i])   # except the i-th column of minus_d_of_row
        numerator = np.exp(minus_d_squared_of_row[j])   # the j-th column of minus_d_of_row
        p = numerator / denominator
        return p

    def SNE_get_q(self, i, j, distance_matrix):
        d_squared_of_row = (distance_matrix[i, :] ** 2)
        minus_d_squared_of_row = -1 * d_squared_of_row
        denominator = np.sum(np.exp(minus_d_squared_of_row)) - np.exp(minus_d_squared_of_row[i])  # except the i-th column of minus_d_of_row
        numerator = np.exp(minus_d_squared_of_row[j])  # the j-th column of minus_d_of_row
        q = numerator / denominator
        return q

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

    def plot_embedding(self, X, y):
        # X: column-wise
        color_map = plt.cm.jet  #--> hsv, brg (good for S curve), rgb, jet, gist_ncar (good for one blob), tab10, Set1, rainbow, Spectral #--> https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
        plt.scatter(X[0, :], X[1, :], c=y, cmap=color_map, edgecolors='k')
        classes = [str(i) for i in range(len(np.unique(y)))]
        n_classes = len(classes)
        cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels(classes)
        plt.show()

    ###### efficient, backup:
    def fit_transform_good1_BETTER(self, max_iterations=1000, step_checkpoint=20, calculate_again=True):
        if calculate_again == False:
            path_to_save = './output/SNE/'
            Y = self.load_variable(name_of_variable="Y_", path=path_to_save + 'Y/')
            return Y
        Y = np.random.rand(self.n_components, self.n_training_images)  # --> rand in [0,1)
        # save the information at checkpoints:
        path_to_save = './output/SNE/'
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save + 'Y/')
        if self.y is not None:
            self.plot_embedding(X=Y, y=self.y)
        print("Calculating p for all pairs...")
        distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        p_matrix = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("---processing for image " + str(sample_index1))
            sigma = 1 / (2 ** 0.5)
            d_squared_of_row = (distance_matrix_originalSpace[sample_index1, :] ** 2) / (2 * (sigma ** 2))
            # d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, np.where(d_squared_of_row==0))
            d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
            minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
            minus_d_squared_of_row = -1 * d_squared_of_row
            denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
            # if denominator == 0:
            #     print("hiiiiiiii")
            for sample_index2 in range(self.n_training_images):
                if sample_index1 != sample_index2:
                    numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                    p = numerator / denominator
                else:
                    p = 0
                p_matrix[sample_index1, sample_index2] = p
        eta = 0.1
        iteration_index = -1
        cost_iters = np.zeros((step_checkpoint, 1))
        update = 0
        # stuck_in_local_min = False
        while True:
            iteration_index = iteration_index + 1
            #----- update alpha:
            if iteration_index < 250:
                alpha = 0.5   #0.5
            else:
                alpha = 0.8  #0.8
            # if iteration_index < 900:
            #     eta = 0.1
            # else:
            #     alpha = 0.05
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=Y)
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                d_squared_of_row = (distance_matrix_embeddedSpace[sample_index1, :] ** 2)
                # d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, np.where(d_squared_of_row==0))
                d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
                minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
                minus_d_squared_of_row = -1 * d_squared_of_row
                denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
                # if denominator == 0:
                #     print("hiiiiiiii")
                for sample_index2 in range(self.n_training_images):
                    if sample_index1 != sample_index2:
                        numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                        q = numerator / denominator
                    else:
                        q = 0
                    q_matrix[sample_index1, sample_index2] = q
            for sample_index1 in range(self.n_training_images):
                y_i_previousIteration = Y[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    y_j_previousIteration = Y[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix[sample_index1, sample_index2]
                    p_ji = p_matrix[sample_index2, sample_index1]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    q_ji = q_matrix[sample_index2, sample_index1]
                    gradient = gradient + (p_ij - q_ij + p_ji - q_ji) * (y_i_previousIteration - y_j_previousIteration)
                gradient = gradient * 2
                update = - (eta * gradient) + (alpha * update)
                y_i = y_i_previousIteration + update
                Y[:, sample_index1] = y_i.ravel()
            #--- add some jitter:
            if iteration_index < 50:
                for sample_index in range(self.n_training_images):
                    noise = np.random.normal(0, 0.1, self.n_components)
                    Y[:, sample_index] = Y[:, sample_index] + noise
            # if stuck_in_local_min == True:
            #     print("Escaping the local optima...")
            #     stuck_in_local_min = False
            #     for sample_index in range(self.n_training_images):
            #         noise = np.random.normal(0, 1, self.n_components)
            #         Y[:, sample_index] = Y[:, sample_index] + noise
            #--- calculate cost:
            cost = 0
            for sample_index1 in range(self.n_training_images):
                for sample_index2 in range(self.n_training_images):
                    if sample_index2 != sample_index1:
                        p_ij = p_matrix[sample_index1, sample_index2]
                        q_ij = q_matrix[sample_index1, sample_index2]
                        # if q_ij != 0:
                            # if (p_ij / q_ij) != 0:
                            #     cost = cost + (p_ij * np.log10(p_ij / q_ij))
                        if p_ij != 0 and q_ij != 0:
                            cost = cost + (p_ij * np.log10(p_ij)) - (p_ij * np.log10(q_ij))
            print("---- cost of this iteration: " + str(cost))
            index_to_save = iteration_index % step_checkpoint
            cost_iters[index_to_save] = cost
            # save the information at checkpoints:
            if (iteration_index+1) % step_checkpoint == 0:
                path_to_save = './output/SNE/'
                print("Saving the checkpoint in iteration #" + str(iteration_index))
                checkpoint_index = int(np.floor(iteration_index / step_checkpoint))
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_variable(variable=Y, name_of_variable="Y_"+str(checkpoint_index), path_to_save=path_to_save+'Y/')
                if self.y is not None:
                    self.plot_embedding(X=Y, y=self.y)
            # # check if trapped in local optima:
            # if (iteration_index + 1) % step_checkpoint == 0:
            #     cost_iters_std = np.std(cost_iters)
            #     if cost_iters_std < 1.0:
            #         stuck_in_local_min = True
            # --- check terminate:
            if max_iterations is not None:
                if iteration_index > max_iterations:
                    return Y

    ###### efficient, backup:
    def fit_transform_good2(self, calculate_again=True):
        path_to_save = './output/SNE/'
        if calculate_again == False:
            Y = self.load_variable(name_of_variable="Y_", path=path_to_save + 'Y/')
            return Y
        Y = np.random.rand(self.n_components, self.n_training_images)  # --> rand in [0,1)
        # save the information at checkpoints:
        self.save_variable(variable=Y, name_of_variable="Y_initial", path_to_save=path_to_save + 'Y/')
        if self.y is not None:
            self.plot_embedding(X=Y, y=self.y)
        print("Calculating p for all pairs...")
        distance_matrix_originalSpace = self.get_distances_btw_points(data_matrix=self.X)
        p_matrix = np.zeros((self.n_training_images, self.n_training_images))
        for sample_index1 in range(self.n_training_images):
            print("---processing for image " + str(sample_index1))
            sigma = 1 / (2 ** 0.5)
            for sample_index2 in range(sample_index1, self.n_training_images):
                if sample_index1 != sample_index2:
                    #---> p_{i|j}:
                    d_squared_of_row = (distance_matrix_originalSpace[sample_index1, :] ** 2) / (2 * (sigma ** 2))
                    d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, sample_index1)   # remove the sample_index1-th column of minus_d_of_row
                    minus_d_squared_of_row_diagonalElementRemoved = -1 * d_squared_of_row_diagonalElementRemoved
                    minus_d_squared_of_row = -1 * d_squared_of_row
                    denominator = np.sum(np.exp(minus_d_squared_of_row_diagonalElementRemoved))
                    numerator = np.exp(minus_d_squared_of_row[sample_index2])  # the sample_index2-th column of minus_d_of_row
                    p = numerator / denominator
                else:
                    p = 0
                p_matrix[sample_index1, sample_index2] = p
        eta = 0.1
        iteration_index = -1
        cost_iters = np.zeros((self.step_checkpoint, 1))
        update = 0
        while True:
            iteration_index = iteration_index + 1
            #----- update alpha:
            if iteration_index < 250:
                alpha = 0.5   #0.5
            else:
                alpha = 0.8  #0.8
            print("Iteration " + str(iteration_index) + "...")
            distance_matrix_embeddedSpace = self.get_distances_btw_points(data_matrix=Y)
            q_matrix = np.zeros((self.n_training_images, self.n_training_images))
            for sample_index1 in range(self.n_training_images):
                d_squared_of_row = (distance_matrix_embeddedSpace[sample_index1, :] ** 2)
                # d_squared_of_row_diagonalElementRemoved = np.delete(d_squared_of_row, np.where(d_squared_of_row==0))
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
                y_i_previousIteration = Y[:, sample_index1].reshape((-1, 1))
                gradient = np.zeros((self.n_components, 1))
                for sample_index2 in range(self.n_training_images):
                    y_j_previousIteration = Y[:, sample_index2].reshape((-1, 1))
                    p_ij = p_matrix[sample_index1, sample_index2]
                    p_ji = p_matrix[sample_index2, sample_index1]
                    q_ij = q_matrix[sample_index1, sample_index2]
                    q_ji = q_matrix[sample_index2, sample_index1]
                    gradient = gradient + (p_ij - q_ij + p_ji - q_ji) * (y_i_previousIteration - y_j_previousIteration)
                gradient = gradient * 2
                update = - (eta * gradient) + (alpha * update)
                y_i = y_i_previousIteration + update
                Y[:, sample_index1] = y_i.ravel()
            #--- add some jitter:
            if iteration_index < 50:
                for sample_index in range(self.n_training_images):
                    noise = np.random.normal(0, 0.1, self.n_components)
                    Y[:, sample_index] = Y[:, sample_index] + noise
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
                self.save_variable(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_np_array_to_txt(variable=cost_iters, name_of_variable="cost_iters_"+str(checkpoint_index), path_to_save=path_to_save+'cost/')
                self.save_variable(variable=Y, name_of_variable="Y_"+str(checkpoint_index), path_to_save=path_to_save+'Y/')
                if self.y is not None:
                    self.plot_embedding(X=Y, y=self.y)
            # --- check terminate:
            if self.max_iterations is not None:
                if iteration_index > self.max_iterations:
                    return Y