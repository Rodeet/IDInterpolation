import numpy as np
from scipy.integrate import quad
import sympy as sp

class IntegralInterpolator:
    def __init__(self, x, y, I):
        self.x = x
        self.y = y
        self.I = I

    def __call__(self, x0):
        interpolated_values = []
        # Находим ближайшие точки в массиве x
        idx = self.find_nearest(x0)

        # Если точка находится за пределами диапазона x
        if idx is None:
            raise ValueError(f"Точка {x0} находится за пределами диапазона x")

        # Если точка совпадает с одной из точек в x
        if x0 in self.x:
            interpolated_values.append(self.y[np.where(self.x == x0)][0])
        else:
            # Интерполяция между ближайшими точками
            x1 = self.x[idx]
            x2 = self.x[idx+1]
            y1 = self.y[idx]
            y2 = self.y[idx+1]
            # Значение интеграла
            I_i = self.I[idx]
            interpolated_values.append(self.lagrange_integral_interp(I_i, x1, x2, y1, y2, x0))
            interpolated_values.append(self.polinom_integral_interp(I_i, x1, x2, y1, y2, x0))

        return interpolated_values

    # Поиск окна интерполяции
    def find_nearest(self, value):
        idx = np.where((self.x <= value) & (np.roll(self.x, -1) > value))[0]
        if len(idx) == 0:
            return None
        return idx[0]

    @staticmethod
    def lagrange_integral_interp(I, x1, x2, y1, y2, x0):
        h = x2 - x1
        # Вычисляем фазу интерполяции
        u = (x0 - x1)/h

        # Вычисляем коэффициенты Лагранжа
        P_I = 6*u*(1-u)
        P_i = (1-u)*(1-3*u)
        P_ip1 = u*(3*u-2)

        # Вычислим параболический интегрально-функциональный многочлен в Лагранжевой форме
        S = (1/h) * P_I*I + P_i*y1 + P_ip1*y2
        return S
    
    @staticmethod
    def polinom_integral_interp(I, x1, x2, y1, y2, x0):
        h = x2 - x1

        df = y2 - y1
        dI = I - h*y1

        # Вычислим параболический интегрально-функциональный многочлен в полиномиальной форме
        a0 = y1
        a1 = (dI*6/h**2) - (2*df/h)
        a2 = (-6*dI/h**3) + (3*df/h**2)

        def S(x):
            return a0 + a1*(x-x1) + a2*(x-x1)**2

        return S(x0)
    
class DiffIntegralInterpolator:
    def __init__(self, x, y, I, d):
        self.x = x
        self.y = y
        self.I = I
        self.d = d

    def __call__(self, x0):
        interpolated_value = 0
        # Находим ближайшие точки в массиве x
        idx = self.find_nearest(x0)

        # Вычисление оценки второй производной
        second_derivative = np.diff(self.d) / np.diff(self.x)
        second_derivative = np.concatenate(([0], second_derivative))
        # Вычисление оценки третьей производной
        third_derivative = np.diff(second_derivative) / np.diff(self.x)

        # Если точка находится за пределами диапазона x
        if idx is None:
            raise ValueError(f"Точка {x0} находится за пределами диапазона x")

        # Если точка совпадает с одной из точек в x
        if x0 in self.x:
            interpolated_value = self.y[np.where(self.x == x0)][0]
        else:
            # Интерполяция между ближайшими точками
            x1 = self.x[idx]
            x2 = self.x[idx+1]
            y1 = self.y[idx]
            y2 = self.y[idx+1]
            # Значение интеграла
            I_i = self.I[idx]
            # Значение первой и третьей производной
            d3 = third_derivative[idx]
            df = self.d[idx+1] - self.d[idx]
            f_i = self.d[idx]
            interpolated_value = self.parabolic_integral_diff_interp(I_i, f_i, df, d3, x1, x2, y1, y2, x0)

        return interpolated_value

    # Поиск окна интерполяции
    def find_nearest(self, value):
        idx = np.where((self.x <= value) & (np.roll(self.x, -1) > value))[0]
        if len(idx) == 0:
            return None
        return idx[0]

    @staticmethod
    def parabolic_integral_diff_interp(I, f_i, df, d3, x1, x2, y1, y2, x0):
        h = x2 - x1
        # Вычисляем фазу интерполяции
        u = (x0 - x1)/h

        delta = h**3 * d3 / (72 * (3**0.5)) # Оценка погрешности

        # Вычислим коэффициенты многочлена S
        a0 = I/h - f_i*h/2 - df*h/6
        a1 = f_i
        a2 = df/(2*h)

        def S(x):   
            return a0 + a1*(x-x1) + a2*(x-x1)**2

        return [S(x0) - delta, S(x0) + delta]