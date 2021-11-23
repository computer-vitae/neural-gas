from neural_gas import NeuralGas
from generator import Generator


def main():
    g = Generator(obs_num=50)
    means = [[2,1],
        [20, 17],
        [8, 9]]
    sigmas = [2, 2, 2]
    data = g.gen(means, sigmas)

    NG = NeuralGas(3, 2, 0.1, 0.5)
    NG.setup(data)
    NG.train(data, 100)


if __name__ == "__main__":
    main()