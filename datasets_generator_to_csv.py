import numpy as np
import pandas as pd


################################################################
########## można wymyślić jeszcze jakiś dataset ################
class Datasets:
    def __init__(self):
        ...

    def to_csv(self, dataframe, name):
        return dataframe.to_csv(name)

    def ball_points_generator(self, dim, N, radius):
        # Generowanie macierzy zmiennych losowych o rozkładzie normalnym standardowym
        xi = np.random.normal(size=(dim, N))

        # obliczanie normy
        norm = np.linalg.norm(xi, axis=0)

        S = xi / norm
        U = np.random.uniform(low=0.0, high=1.0, size=N)
        S = S * np.power(U, 1 / dim)
        points = S * radius

        # Transponowanie macierzy punktów
        # points = scaled_xi.T

        return points.T

    # %%
    def sphere_points_generator(self, dim, N, radius):
        # Generowanie macierzy zmiennych losowych o rozkładzie normalnym standardowym
        xi = np.random.normal(size=(dim, N))

        # Obliczanie sumy kwadratów zmiennych losowych dla każdego punktu
        sum_of_squares = np.sum(xi ** 2, axis=0)

        # Obliczanie wartości λ na podstawie żądanej wartości
        lambda_squared = radius ** 2

        # Skalowanie zmiennych losowych dla każdego punktu, aby uzyskać żądaną wartość λ
        scaled_xi = xi * np.sqrt(lambda_squared / sum_of_squares)

        # Transponowanie macierzy punktów
        # unit_vectors = scaled_xi.T

        # return unit_vecto
        return scaled_xi.T

    ## kod pożyczony
    # (można sprawdzić, czy napewno dobry w wielu wymiarach)
    # torus
    def generate_n_dimensional_torus(self, n_dimensions, n_points, R=2, r=1):
        """
        Creates an n-dimensional torus.

        Parameters:
        n_dimensions : int
            The number of dimensions for the torus. Must be greater than or equal to 2.
        n_points : int
            The number of points.
        R : float
            The distance from the center of the tube to the center of the torus.
        r : float
            The radius of the tube (distance from the center of the tube to the torus surface).

        Returns:
        numpy.ndarray
            The n-dimensional points of the torus.
        """

        assert n_dimensions >= 2, "Number of dimensions must be greater than or equal to 2."

        # generate n_points random angles for each dimension
        angles = np.random.uniform(0, 2 * np.pi, (n_points, n_dimensions))

        # calculate the n-dimensional points on the torus
        coordinates = []
        for i in range(n_dimensions):
            if i == 0:
                coordinate = (R + r * np.cos(angles[:, 1])) * np.cos(angles[:, 0])
            elif i == 1:
                coordinate = (R + r * np.cos(angles[:, 1])) * np.sin(angles[:, 0])
            else:
                coordinate = r * np.sin(angles[:, i])
            coordinates.append(coordinate)

        return np.array(coordinates).T

    def one_sphere_one_ball_inside_dataset(self, dim, points_number, sphere_r=4, ball_r=2):
        sphere_points = self.sphere_points_generator(dim, points_number, sphere_r)
        ball_points = self.ball_points_generator(dim, points_number, ball_r)

        data = np.concatenate((sphere_points, ball_points))

        y1 = np.zeros(points_number)
        y2 = np.ones(points_number)
        labels = np.concatenate((y1, y2))
        df = pd.DataFrame(data)
        df['labels'] = labels

        return self.to_csv(df, f'datasets/one_sphere_one_ball_inside/{dim}_points_{points_number}.csv')

    def two_sphere_one_ball_inside_dataset(self, dim, points_number, sphere1_r=4, sphere2_r=4, ball_r=2):
        sphere_points = self.sphere_points_generator(dim, points_number, sphere1_r)
        sphere_points2 = self.sphere_points_generator(dim, points_number, sphere2_r)
        ball_points = self.ball_points_generator(dim, points_number, ball_r)

        data = np.concatenate((sphere_points, sphere_points2, ball_points))

        y1 = np.zeros(points_number)
        y2 = np.ones(points_number)
        y3 = np.full(points_number, 2)
        labels = np.concatenate((y1, y2, y3))
        df = pd.DataFrame(data)
        df['labels'] = labels

        return self.to_csv(df, f'datasets/two_sphere_one_ball_inside/{dim}_points_{points_number}.csv')

    def one_sphere_one_torus_inside_dataset(self, dim, points_number, sphere_r=4, torus_R=2, torus_r=1):

        sphere_points = self.sphere_points_generator(dim, points_number, sphere_r)
        torus_points = self.generate_n_dimensional_torus(dim, points_number, torus_R, torus_r)

        data = np.concatenate((sphere_points, torus_points))

        y1 = np.zeros(points_number)
        y2 = np.ones(points_number)
        labels = np.concatenate((y1, y2))
        df = pd.DataFrame(data)
        df['labels'] = labels

        return self.to_csv(df, f'datasets/one_sphere_one_torus_inside/dim_{dim}_points_{points_number}.csv')


datasets = Datasets()

points_list = [1000, 2000, 5000, 20000, 50000, 100000]
dimmentions = [3, 5, 10, 20, 50]

# points_list = [1000, 2000]
# dimmentions = [3, 5]


for points in points_list:
    for dim in dimmentions:
        datasets.one_sphere_one_torus_inside_dataset(dim, points)
        datasets.two_sphere_one_ball_inside_dataset(dim, points)
        datasets.one_sphere_one_ball_inside_dataset(dim, points)

