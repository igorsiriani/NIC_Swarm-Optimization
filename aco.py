import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from matplotlib.animation import FuncAnimation
import utils as u
import matplotlib.pyplot as plt


class ACO:
    # implementação do ACO
    def __init__(self, distance, n_ants=1, n_elitists=1, n_iterations=10, alpha=1, beta=5, decay=0.5):
        self.nodes_df = None
        self.anim = None
        self.pbest_plot = None
        self.best_path = None
        self.short_path = None
        self.D = distance

        self.n_ants = n_ants
        self.n_elitists = n_elitists
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.it_current = 0
        self.it_best = 0

        self.pheromones = np.ones(self.D.shape) / len(self.D)
        self.fig, self.ax = plt.subplots()

    # calcula distâncias entre caminhos
    def generate_path_distance(self, paths):
        # distância total 0
        total = 0

        # para cada caminho
        for path in paths:
            # adiciona distância ao total
            total += self.D[path]

        return total

    # gera caminho
    def generate_path(self, start):
        path = []
        visited = set()

        # adiciona início ao caminho
        visited.add(start)
        # retoma caminho anterior
        previous = start

        # para cada distância
        for _ in range(len(self.D) - 1):
            # calcula o movimento
            current = self.choose_path(
                self.pheromones[previous], self.D[previous], visited)
            # acrescenta caminho encontrado
            path.append((previous, current))
            previous = current
            visited.add(current)

        # volta ao início
        path.append((previous, start))

        return path

    # gera todos os caminhos
    def generate_all_paths(self):
        paths = []

        # para cada formiga
        for _ in range(self.n_ants):
            # gera um caminho do início
            path = self.generate_path(0)
            # adiciona a distância
            paths.append((path, self.generate_path_distance(path)))

        return paths

    # escolhe caminho
    def choose_path(self, pheromone, distance, visited):
        p = np.copy(pheromone)

        # zera feromônio para cidades visitadas
        p[list(visited)] = 0
        # calcula feromônios e probabilidade de movimento
        track = p ** self.alpha * ((1 / distance) ** self.beta)
        track_norm = track / track.sum()
        index = np.random.choice(range(len(self.D)), 1, p=track_norm)[0]

        return index

    # espalha feromônio
    def spread_pheromone(self, paths):
        # sort todos caminhos
        sort_paths = sorted(paths, key=lambda x: x[1])

        # para cada caminho e cada formiga da elite
        for path, _ in sort_paths[:self.n_elitists]:
            for p in path:
                # espalha feromônio
                self.pheromones[p] += 1 / self.D[p]

    # adiciona update ao plot da animação
    def animate(self, i):
        if (self.it_current + 1) <= self.n_iterations:
            title = 'Iteration {:02d}'.format(i)
            self.run()

            nodes = pd.DataFrame(self.best_path[0], columns=['order', 'next'])
            nodes_df2 = self.nodes_df.reindex(nodes['order'])
            nodes_df2 = pd.concat([nodes_df2, nodes_df2.head(1)], ignore_index=True)

            # atualiza imagem
            self.ax.set_title(title)
            self.pbest_plot.set_xdata(nodes_df2['latitude'])
            self.pbest_plot.set_ydata(nodes_df2['longitude'])
        else:
            self.anim.event_source.stop()

    def aco(self, nodes):
        # inicializa variáveis
        self.nodes_df = nodes
        self.short_path = None
        self.best_path = ('', np.inf)

        # inicializa figura
        self.fig.set_tight_layout(True)
        self.pbest_plot, = self.ax.plot(nodes['latitude'], nodes['longitude'], marker='o', color='black', alpha=0.5)

        self.anim = FuncAnimation(self.fig, self.animate, frames=list(range(1, self.n_iterations)), interval=50, blit=False,
                             repeat=True)
        self.anim.save("gif/ACO.gif", dpi=120, writer="imagemagick")

        return self.best_path[1], self.best_path[0], self.it_best

    # função que faz a iteração
    def run(self):
        self.it_current += 1
        # gera os caminhos
        paths = self.generate_all_paths()
        # espalha feromônio
        self.spread_pheromone(paths)
        # calcula caminho mais curto
        self.short_path = min(paths, key=lambda x: x[1])
        # avalia o menor custo
        if self.short_path[1] < self.best_path[1]:
            self.best_path = self.short_path
            self.it_best = self.it_current

        # decaimento do feromônio
        self.pheromones *= self.decay


def main():
    # Declara inputs
    input_file = 'data/berlin52.tsp'
    nodes = u.read_tsp(input_file)

    # inicia variáveis
    D = u.create_distances(nodes)
    ants = 52
    elitists = 10
    iterations = 200
    alpha = 1
    beta = 2
    decay = 0.8

    result_list = []
    it_list = []
    for i in range(0, 100):
        # instancia o ACO
        a = ACO(D, n_ants=ants, n_elitists=elitists, n_iterations=iterations, alpha=alpha, beta=beta, decay=decay)

        nodes_df = pd.DataFrame(list(map(np.ravel, nodes)), columns=['latitude', 'longitude'])
        cost, path, it_best = a.aco(nodes_df)
        print(f'Best iteration: {it_best} | Cost: {cost}\n')
        result_list.append(cost)
        it_list.append(it_best)
        a = None

    print('Custo mínimo: ', min(result_list))
    print('Custo máximo: ', max(result_list))
    print('Custo médio: ', np.mean(result_list))
    print('Custo padrão: ', np.std(result_list))

    print('Iteração mínima: ', min(it_list))
    print('Iteração máxima: ', max(it_list))
    print('Iteração média: ', np.mean(it_list))
    print('Iteração padrão: ', np.std(it_list))


if __name__ == "__main__":
    main()
