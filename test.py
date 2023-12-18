from IDInterpolationMethods import IntegralInterpolator, DiffIntegralInterpolator
import numpy as np

# Сеточная функция
x_values = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
y_values = np.array([-64, -27, -8, -1, 0, 1, 8, 27, 64])
# Значения интеграла в точке
I_values = np.array([-43.75, -16.25, -3.75, -0.25, 0.25, 3.75, 16.25, 43.75, 92.25])
x0 = 2.5

# Создание объекта параболического интегрального интерполятора
p_i_interp = IntegralInterpolator(x_values, y_values, I_values)
# Получение приближенных значений
approximated_value = p_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value}")

# Значения производной в точке
d_values = np.array([48, 27, 12, 3, 0, 3, 12, 27, 48])

# Создание объекта дифференциально-интегрального интерполятора
p_d_i_interp = DiffIntegralInterpolator(x_values, y_values, I_values, d_values)
# Получение приближенных значений
approximated_value_2 = p_d_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value_2}")

x0 = -2.5
approximated_value = p_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value}")
approximated_value_2 = p_d_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value_2}")

x0 = 2
approximated_value = p_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value}")
approximated_value_2 = p_d_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value_2}")

x0 = -2
approximated_value = p_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value}")
approximated_value_2 = p_d_i_interp(x0)
print(f"Приближенное значение в точке {x0} равно {approximated_value_2}")