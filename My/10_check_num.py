def check_num_in(big_num, find_num):
    big_num_str = str(big_num)
    find_num_str = str(find_num)

    return find_num_str in list(big_num_str)

def check_num_in(big_num, find_num):
    big_num_str_list = list(str(big_num))
    big_num_list = list(map(int, big_num_str_list))

    return find_num in big_num_list

print(check_num_in(12345678, 1))


def check_3_and_7_in_num(big_num):
    """
    Дано n-значное целое число N. Определите, входят ли в него цифры 3 и 7.
    Напишите соответствующее выражение для проверки.

    :param big_num:
    :return:
    """
    big_num_str_list = list(str(big_num))

    return '3' in big_num_str_list and '7' in big_num_str_list


print(check_3_and_7_in_num(1245678))


def check_num_is_begin_from_even_num(big_num):
    """
    Дано n-значное целое число N. Определите, начинается ли оно с чётной цифры.
    Напишите соответствующее выражение для проверки.

    :param big_num:
    :return:
    """
    big_num_str_list = list(str(big_num))
    first_num = int(big_num_str_list[0])
    print(first_num)
    return first_num % 2 == 0


print(check_num_is_begin_from_even_num(1613123))

list_1 = [1, 2]

list_2 = [1, 2, 3]
val = list_2.pop()

print(list_1 == list_2)

b = 10

b /= b
print(b)