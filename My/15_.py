import random

my_list = []
for i in range(0, 10000):
    my_list.append(random.randint(0, 1))

num_dict = {}

for num in my_list:
    if num not in num_dict:
        num_dict[num] = 1
    else:
        num_dict[num] += 1


print(num_dict)


print("************")
# Задайте список my_list, содержащий целые числа, с помощью модуля random, как и в предыдущей задаче.
#
# Используя инструкцию while, разработайте программу,
# которая вычисляет сумму элементов списка, пока она не превысит число 10.
#
# В результате работы программы выведите значение полученной суммы
# (обратите внимание: так как числа в списке my_list являются случайными,
# то значение суммы при каждом запуске программы может быть разным).


my_num_list = []

for i in range(10):
    my_num_list.append(random.randint(0, 9))

i = 0
sum = 0
while sum < 10:
    sum += my_num_list[i]
    i += 1

print(sum)