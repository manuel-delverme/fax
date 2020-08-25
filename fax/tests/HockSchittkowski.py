from jax.numpy import *


class Hs:
    constraints = lambda x: zeros_like(x)


class Hs01(Hs):
    @staticmethod
    def initialize():
        return zeros(2)  # x

    _objective_function = lambda x: -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)


class Hs06(Hs):
    @staticmethod
    def initialize():
        return zeros(2)  # x

    _objective_function = lambda x: -((1 - x[0]) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: 10 * (x[1] - x[0] ** 2) - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs07(Hs):
    @staticmethod
    def initialize():
        return zeros(2)  # x

    _objective_function = lambda x: -(log(1 + x[0] ** 2) - x[1])

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-sqrt(3))
    h0 = lambda x: (1 + x[0] ** 2) ** 2 + x[1] ** 2

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs08(Hs):
    @staticmethod
    def initialize():
        return zeros(2)  # x

    _objective_function = lambda x: -(-1)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-1)
    h0 = lambda x: x[0] ** 2 + x[1] ** 2
    h1 = lambda x: x[0] * x[1]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs09(Hs):
    @staticmethod
    def initialize():
        return zeros(2)  # x

    _objective_function = lambda x: -(sin(pi * x[0] / 12) * cos(pi * x[1] / 16))

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-0.5)
    h0 = lambda x: 4 * x[0] - 3 * x[1] - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs26(Hs):
    @staticmethod
    def initialize():
        return zeros(3)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: (1 + x[1] ** 2) * x[0] + x[2] ** 4

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs27(Hs):
    @staticmethod
    def initialize():
        return zeros(3)  # x

    _objective_function = lambda x: -((x[0] - 1) ** 2 / 100 + (x[1] - x[0] ** 2) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0.04)
    h0 = lambda x: x[0] + x[2] ** 2

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs28(Hs):
    @staticmethod
    def initialize():
        return zeros(3)  # x

    _objective_function = lambda x: -((x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x),))


class Hs39(Hs):
    @staticmethod
    def initialize():
        return zeros(4)  # x

    _objective_function = lambda x: -(-x[0])

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-1)
    h0 = lambda x: x[1] - x[0] ** 3 - x[2] ** 2 - 0
    h1 = lambda x: x[0] ** 2 - x[1] - x[3] ** 2 - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs40(Hs):
    @staticmethod
    def initialize():
        return zeros(4)  # x

    _objective_function = lambda x: -(-x[0] * x[1] * x[2] * x[3])

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-0.25)
    h0 = lambda x: x[0] ** 3 + x[1] ** 2
    h1 = lambda x: x[0] ** 2 * x[3] - x[2] - 0
    h2 = lambda x: x[3] ** 2 - x[1] - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs46(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] ** 2 * x[3] + sin(x[3] - x[4])
    h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs47(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 3 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
    h1 = lambda x: x[1] - x[2] ** 2 + x[3]
    h2 = lambda x: x[0] * x[4]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs49(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + 3 * x[3]
    h1 = lambda x: x[2] + 5 * x[4]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs50(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]
    h1 = lambda x: x[1] + 2 * x[2] + 3 * x[3]
    h2 = lambda x: x[2] + 2 * x[3] + 3 * x[4]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs51(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 3 * x[1]
    h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
    h2 = lambda x: x[1] - x[4] - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs52(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((4 * x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(1859 / 349)
    h0 = lambda x: x[0] + 3 * x[1] - 0
    h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
    h2 = lambda x: x[1] - x[4] - 0

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs61(Hs):
    @staticmethod
    def initialize():
        return zeros(3)  # x

    _objective_function = lambda x: -(4 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2 - 33 * x[0] + 16 * x[1] - 24 * x[2])

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(- 143.6461422)
    h0 = lambda x: 3 * x[0] - 2 * x[1] ** 2
    h1 = lambda x: 4 * x[0] - x[2] ** 2

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs77(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0.24150513)
    h0 = lambda x: x[0] ** 2 * x[3] + sin(x[3] - x[4])
    h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x),))


class Hs78(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -(x[0] * x[1] * x[2] * x[3] * x[4])

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(-2.91970041)
    h0 = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2
    h1 = lambda x: x[1] * x[2] - 5 * x[3] * x[4] - 0
    h2 = lambda x: x[0] ** 3 + x[1] ** 3

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))


class Hs79(Hs):
    @staticmethod
    def initialize():
        return zeros(5)  # x

    _objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)

    @classmethod
    def objective_function(cls, x):
        return cls._objective_function(x)

    optimal_solution = -array(0.0787768209)
    h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
    h1 = lambda x: x[1] - x[2] ** 2 + x[3]
    h2 = lambda x: x[0] * x[4]

    @classmethod
    def constraints(cls, x):
        return stack((cls.h0(x), cls.h1(x), cls.h2(x),))
