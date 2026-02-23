import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from typing import Tuple

from util import create_path

class Plotter():

    def __init__(self, bounds, num_dims: int = 1, resolution: int = 20, path: str = "./"):
    
        self.num_dims = num_dims
        self.bounds = bounds
        self.path = path
        self.resolution = resolution

        create_path(path)

        if self.num_dims == 1:
            self.x = np.linspace(self.bounds[0, 0], self.bounds[1, 0], self.resolution)
        elif self.num_dims == 2:
            self.x1 = torch.linspace(self.bounds[0, 0], self.bounds[1, 0], self.resolution)
            self.x2 = torch.linspace(self.bounds[0, 1], self.bounds[1, 1], self.resolution)

            self.grid_x1, self.grid_x2 = torch.meshgrid(self.x1, self.x2, indexing='xy')
            self.grid = torch.stack([self.grid_x1.reshape(-1), self.grid_x2.reshape(-1)], dim=1)

            self.grid_x1 = self.grid_x1.numpy()
            self.grid_x2 = self.grid_x2.numpy()
        else:
            raise ValueError("Plotter is only prepared for 1 and 2 dimensional problems")

    def generate_figure_acquisition_function_2D(self, observations, acquisition_function, file_name: str, title_figure: str, color_cmap: str = "Reds", figsize: Tuple = (16, 12)):

        values_acq = acquisition_function(self.grid[ :, None ])
        values_acq = values_acq.detach().numpy()

        plt.figure(figsize=figsize)
        contour_acq = plt.contourf(self.grid_x1, self.grid_x2, values_acq.reshape(self.grid_x1.shape), cmap='viridis', levels=50)
        plt.colorbar(contour_acq, label=r"$\alpha(x)$")
        i_max = values_acq.argmax()
        plt.scatter(self.grid_x1.flatten()[ i_max ], self.grid_x2.flatten()[ i_max ], color='blue', s=100, marker='X', label="Max")
        
        colors = np.linspace(0, 1, observations.shape[ 0 ])
        colormap = plt.colormaps.get_cmap(color_cmap)
        point_colors = [colormap(color) for color in colors]
        plt.scatter(observations[ : , 0 ], observations[ : , 1 ], color=point_colors, marker='x', s=100, label="Observations")
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.legend(fontsize=15)
        plt.title(title_figure, fontsize=15)
        plt.savefig(f"{self.path}/{file_name}.pdf", format='pdf', dpi=1000)
        plt.close()

    def generate_figure_2D_problem(self, observations, values_obs, problem, x_max, file_name: str, title_figure: str, color_cmap: str = "Reds", figsize: Tuple = (16, 12)):
                
        values_problem = problem(self.grid)
        values_problem = values_problem.detach().numpy()
        plt.figure(figsize=figsize)
        plt.contourf(self.grid_x1, self.grid_x2, values_problem.reshape(self.grid_x1.shape), cmap='viridis', levels=50)
        plt.scatter(x_max[ 0, 0 ], x_max[ 0, 1 ], color='blue', s=100, marker='X', label="Max")
        
        colors = np.linspace(0, 1, observations.shape[ 0 ])
        colormap = plt.colormaps.get_cmap(color_cmap)
        point_colors = [colormap(color) for color in colors]
        plt.scatter(observations[ : , 0 ], observations[ : , 1 ], color=point_colors, marker='x', s=100, label="Observations")
        plt.xlabel(r"$x_{1}$", fontsize=16)
        plt.ylabel(r"$x_{2}$", fontsize=16)
        plt.legend(fontsize=15)
        plt.title(title_figure, fontsize=15)
        plt.savefig(f"{self.path}/{file_name}.pdf", format='pdf', dpi=1000)
        plt.close()
        
    def generate_figure_regret(self, acquisition_name, num_iters, log_rel_diffs, title_figure, file_name):

        plt.plot(np.arange(1, num_iters + 1), log_rel_diffs.detach().numpy(), label=acquisition_name)
        plt.legend(fontsize=18, ncol=2)
        plt.xlabel(r"Evaluations", fontsize=20)
        plt.ylabel(r"Log. Rel. Diff.", fontsize=20)
        plt.title(title_figure, fontsize=20)
        plt.savefig(f"{self.path}/{file_name}.pdf", format='pdf', dpi=1000)
        plt.close()
        
