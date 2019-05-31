import numpy as np
import itertools


class SOM(object):

    def __init__(self, h, w, dim_feat):
        """
            Construction of a zero-filled SOM.
            h,w,dim_feat: constructs a (h,w,dim_feat) SOM.
        """
        self.shape = (h, w, dim_feat)
        self.som = np.zeros((h, w, dim_feat))

        # Training parameters
        self.L0 = 0.0
        self.lam = 0.0
        self.sigma0 = 0.0

        self.data = []

        self.hit_score = np.zeros((h, w))

    def train(self, data, L0, lam, sigma0, initializer=np.random.rand, frames=None):
        """
            Training procedure for a SOM.
            data: a N*d matrix, N the number of examples,
                  d the same as dim_feat=self.shape[2].
            L0,lam,sigma0: training parameters.
            initializer: a function taking h,w and dim_feat (*self.shape) as
                         parameters and returning an initial (h,w,dim_feat) tensor.
            frames: saves intermediate frames if not None.
        """
        self.L0 = L0
        self.lam = lam
        self.sigma0 = sigma0

        self.som = initializer(*self.shape)

        self.data = data

        for t in itertools.count():
            if frames != None:
                frames.append(self.som.copy())

            if self.sigma(t) < 0.5:
                print("final t:", t)
                # print("quantization error:", self.quant_err())
                break

            i_data = np.random.choice(range(len(data)))

            bmu = self.find_bmu(data[i_data])
            self.hit_score[bmu] += 1

            self.update_som(bmu, data[i_data], t)

    def quant_err(self):
        """
            Computes the quantization error of the SOM.
            It uses the data fed at last training.
        """
        bmu_dists = []
        for input_vector in self.data:
            bmu = self.find_bmu(input_vector)
            bmu_feat = self.som[bmu]
            bmu_dists.append(np.linalg.norm(input_vector - bmu_feat))
        return np.array(bmu_dists).mean()

    def find_bmu(self, input_vec):
        """
            Find the BMU of a given input vector.
            input_vec: a d=dim_feat=self.shape[2] input vector.
        """
        list_bmu = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist = np.linalg.norm((input_vec - self.som[y, x]))
                list_bmu.append(((y, x), dist))
        list_bmu.sort(key=lambda x: x[1])
        return list_bmu[0][0]

    def update_som(self, bmu, input_vector, t):
        """
            Calls the update rule on each cell.
            bmu: (y,x) BMU's coordinates.
            input_vector: current data vector.
            t: current time.
        """
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist_to_bmu = np.linalg.norm((np.array(bmu) - np.array((y, x))))
                self.update_cell((y, x), dist_to_bmu, input_vector, t)

    def update_cell(self, cell, dist_to_bmu, input_vector, t):
        """
            Computes the update rule on a cell.
            cell: (y,x) cell's coordinates.
            dist_to_bmu: L2 distance from cell to bmu.
            input_vector: current data vector.
            t: current time.
        """
        self.som[cell] += self.N(dist_to_bmu, t) * self.L(t) * (input_vector - self.som[cell])

    def update_bmu(self, bmu, input_vector, t):
        """
            Update rule for the BMU.
            bmu: (y,x) BMU's coordinates.
            input_vector: current data vector.
            t: current time.
        """
        self.som[bmu] += self.L(t) * (input_vector - self.som[bmu])

    def L(self, t):
        """
            Learning rate formula.
            t: current time.
        """
        return self.L0 * np.exp(-t / self.lam)

    def N(self, dist_to_bmu, t):
        """
            Computes the neighbouring penalty.
            dist_to_bmu: L2 distance to bmu.
            t: current time.
        """
        curr_sigma = self.sigma(t)
        return np.exp(-(dist_to_bmu ** 2) / (2 * curr_sigma ** 2))

    def sigma(self, t):
        """
            Neighbouring radius formula.
            t: current time.
        """
        return self.sigma0 * np.exp(-t / self.lam)