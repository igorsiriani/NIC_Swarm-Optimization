import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PSO:

    def __init__(self, n_particles, x_interval, y_interval, w, c1, c2, n_iter):
        self.X = None
        self.V = None
        self.gbest = None
        self.pbest_obj = None
        self.pbest = None
        self.gbest_plot = None
        self.p_arrow = None
        self.gbest_obj = None
        self.p_plot = None
        self.pbest_plot = None
        self.y = None
        self.x = None
        
        self.n_particles = n_particles
        self.x_interval = x_interval
        self.y_interval = y_interval

        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_iter = n_iter

    # função de avaliação
    def evaluate(self, x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def update(self):
        "Function to do one iteration of particle swarm optimization"
        # global 
        # Update params
        r1, r2 = np.random.rand(2)
        self.V = self.w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (self.gbest.reshape(-1, 1) - self.X)
        self.X = self.X + self.V
        obj = self.evaluate(self.X[0], self.X[1])
        self.pbest[:, (self.pbest_obj >= obj)] = self.X[:, (self.pbest_obj >= obj)]
        self.pbest_obj = np.array([self.pbest_obj, obj]).min(axis=0)
        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

    def animate(self, i):
        "Steps of PSO: algorithm update and show in plot"
        title = 'Iteration {:02d}'.format(i)
        # Update params
        self.update()
        # Set picture
        self.ax.set_title(title)
        self.pbest_plot.set_offsets(self.pbest.T)
        self.p_plot.set_offsets(self.X.T)
        self.p_arrow.set_offsets(self.X.T)
        self.p_arrow.set_UVC(self.V[0], self.V[1])
        self.gbest_plot.set_offsets(self.gbest.reshape(1, -1))

    def pso(self):
        # criação de uma população inicial de indivíduos
        x, y = np.array(np.meshgrid(np.linspace(self.x_interval[0], self.x_interval[1], 100), np.linspace(self.y_interval[0], self.y_interval[1], 100)))
        z = self.evaluate(x, y)
    
        # encontre o minimo global
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]
    
        # cria particulas
        np.random.seed(100)
        self.X = np.random.rand(2, self.n_particles) * 5
        self.V = np.random.randn(2, self.n_particles) * 0.1
    
        # Initialize data
        self.pbest = self.X
        self.pbest_obj = self.evaluate(self.X[0], self.X[1])
        self.gbest = self.pbest[:, self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()
    
        # Set up base figure: The contour map
        self.fig.set_tight_layout(True)
        img = self.ax.imshow(z, extent=[self.x_interval[0], self.x_interval[1], self.y_interval[0], self.y_interval[1]], origin='lower', cmap='viridis', alpha=0.5)
        self.fig.colorbar(img, ax=self.ax)
        self.ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
        contours = self.ax.contour(x, y, z, 10, colors='black', alpha=0.4)
        self.ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        self.pbest_plot = self.ax.scatter(self.pbest[0], self.pbest[1], marker='o', color='black', alpha=0.5)
        self.p_plot = self.ax.scatter(self.X[0], self.X[1], marker='o', color='blue', alpha=0.5)
        self.p_arrow = self.ax.quiver(self.X[0], self.X[1], self.V[0], self.V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
        self.gbest_plot = plt.scatter([self.gbest[0]], [self.gbest[1]], marker='*', s=100, color='black', alpha=0.4)
        self.ax.set_xlim([self.x_interval[0], self.x_interval[1]])
        self.ax.set_ylim([self.y_interval[0], self.y_interval[1]])
    
        anim = FuncAnimation(self.fig, self.animate, frames=list(range(1, self.n_iter)), interval=500, blit=False, repeat=True)
        anim.save("PSO.gif", dpi=120, writer="imagemagick")
    
        return self.gbest, self.gbest_obj


def main():
    # inicia variáveis
    c1 = 0.1
    c2 = 0.1
    w = 0.8
    n_particles = 20
    x_interval = [-5, 5]
    y_interval = [-5, 5]
    n_iter = 10

    result_list = []
    for i in range(0, 100):
        swarm = PSO(n_particles, x_interval, y_interval, w, c1, c2, n_iter)
        result, res_obj = swarm.pso()
        print("PSO found best solution at f({})={}".format(result, res_obj))
        result_list.append(result)
    #
    # print('Solução máxima: ', max(result_list))
    # print('Solução mínima: ', min(result_list))
    # print('Solução média: ', np.mean(result_list))
    # print('Solução padrão: ', np.std(result_list))


if __name__ == '__main__':
    main()
