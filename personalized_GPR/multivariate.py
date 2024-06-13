import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessRegression():
    def __init__(self, correlation_length = 1, domine = [-1, 1], prior = None, init_dev = 1):
        self.n_samples_for_line = 100
        self.correlation_length = correlation_length
        self.domine = domine
        self.kernel = self.get_kernel()
        self.data_points = [[],[],[]]
        self.x = np.linspace(self.domine[0],self.domine[1],self.n_samples_for_line)
        self.init_dev = init_dev
        self.prior = prior

    def get_prior_mean(self):
        if self.prior is None:
            def get_m(_x):
                return 0
            return get_m
        else:
            return self.prior
    
    def get_kernel(self):
        def get_k(x1,x2,h):
            return np.exp(-np.sum((x1-x2)**2/h**2))
        return get_k
    
    def sample_lines_and_plot(self, n_lines, show_min=False):

        ms, cov = self.get_mean_and_cov()

        ys = []

        for _ in range(n_lines):
            ys.append(np.random.multivariate_normal(ms,cov))

        for y in ys:
            plt.plot(self.x,y, color="tab:blue", ls="--", alpha=0.8)
            if show_min == True:
                argmin_ = np.argmin(y)
                plt.plot(self.x[argmin_],y[argmin_], color="black", marker="o")
        self.plot_probability_region()
        return
    
    def plot_probability_region(self):

        ms, cov = self.get_mean_and_cov()

        plt.plot(self.x, ms, color="gray")
        plt.fill_between(self.x, ms+0.2*np.diag(cov)**.5, ms-0.2*np.diag(cov)**.5, color="grey", alpha=0.3)
        plt.fill_between(self.x, ms+np.diag(cov)**.5, ms-np.diag(cov)**.5, color="grey", alpha=0.15)
        plt.fill_between(self.x, ms+2*np.diag(cov)**.5, ms-2*np.diag(cov)**.5, color="grey", alpha=0.1)
        plt.plot(self.data_points[0], self.data_points[1], ls="", marker="o")
        
        return
    
    def get_mean_and_cov(self):
        """
            X1 (2D np.array) :  test points (parameters)
            X2 (2D np.array) :  training points (parameters)
            f2 (np.array)    :  training points (costs)
            err (np.array)   :  training cost errors
        """
        X_1 = np.array(self.x)
        X_2 = np.array(self.data_points[0])
        f_2 = np.array(self.data_points[1])
        err = np.array(self.data_points[2])

        prior = self.get_prior_mean()
        prior_means_test = np.array([prior(x_) for x_ in X_1])

        if len(self.data_points[0]) == 0:
            mean = prior_means_test
            cov = np.zeros([self.n_samples_for_line, self.n_samples_for_line])
            for i in range(self.n_samples_for_line):
                for j in range(self.n_samples_for_line):
                    cov[i,j] = self.kernel(X_1[i],X_1[j],self.correlation_length) * self.init_dev**2
        else:
            K_12 = np.zeros([len(X_1), len(X_2)])
            K_22 = np.zeros([len(X_2), len(X_2)])
            K_11 = np.zeros([len(X_1), len(X_1)])

            for i in range(len(X_1)):
                for j in range(len(X_1)):
                    K_11[i,j] = self.kernel(X_1[i], X_1[j], self.correlation_length) * self.init_dev**2
                    
            for i in range(len(X_2)):
                for l in range(len(X_1)):
                    K_12[l,i] = self.kernel(X_1[l], X_2[i], self.correlation_length) * self.init_dev**2

                for j in range(len(X_2)):
                    K_22[i,j] = self.kernel(X_2[i], X_2[j], self.correlation_length) * self.init_dev**2
                    if i == j:
                        K_22[i,i] = K_22[i,i] + err[i]**2 * self.init_dev**2
            
            prior_means_training = np.array([prior(x_) for x_ in X_2])
            mean = prior_means_test+K_12.dot(np.linalg.inv(K_22).dot(f_2-prior_means_training))
            cov = K_11 - K_12.dot(np.linalg.inv(K_22).dot(K_12.transpose()))

        return mean, cov

    def sample_minarg(self, plot = False):
        
        ms, cov = self.get_mean_and_cov()
        
        y = np.random.multivariate_normal(ms,cov)

        minarg_ = np.argmin(y)

        if plot == True:
            plt.plot(self.x,y, color="tab:blue", ls="--", alpha=0.8)
            plt.plot(self.x[minarg_], y[minarg_], color="black", marker="o")

        return np.array([self.x[minarg_]]), np.array([y[minarg_]])

    def add_data_points(self, x_new=[], y_new=[], err_new=[]):

        if len(x_new)==len(y_new) and len(x_new)==len(err_new):
            for i in range(len(x_new)):
                self.data_points[0].append(x_new[i])
                self.data_points[1].append(y_new[i])
                self.data_points[2].append(err_new[i])
        else:
            print("len(x_new) must be equal to len(y_new)")
            raise
        return
