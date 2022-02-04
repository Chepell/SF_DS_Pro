a = 3
b = float(a)
print(b)


num1 = 3
num2 = 3.5
sum1 = num1 + num2

num1 = float(num1)
sum2 = num1 + num2
print(sum1)
print(sum2)

int_one = 1
str_one = str(int_one)
print(str_one)

str_age_begin = 'my age is '
int_age = 18
result_string = str_age_begin + str(int_age)
print(result_string)


str_brothers_begin = 'I have '
count = 5
str_brothers_end = ' brothers'
result_sentence = str_brothers_begin + str(count) + str_brothers_end
print(result_sentence)

str_int = '7'
x = int(str_int)


list1 = list(range(1, 6))
print(tuple(list1))


tp1 = ('a', 'b', 'c', 'd', 'e', )
list2 = list(tp1)
print(list2)


list1 = []

list1.append(1)
list1.append(2)
list1.append(3)
print(list1)

list2 = list(range(-5, 15, 2))
list3 = [1, 2, 3, 4]
list3.extend(list2)
print(list3)

list6 = [3, 1, -10, 5, 11, 20, 1, -10]
list7 = list6.copy()

list6.sort()
list6.reverse()
list_result = list6
print(list7)
print(list_result)


print(4.14 // 2)