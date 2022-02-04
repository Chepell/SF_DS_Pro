matrix = [
    [1, 2],
    [3, 4],
    [5, 6]
]

print(matrix[1][1])

scores = [
    [0.5, 0.6, 0.6, 0.65, 0.3],
    [0.55, 0.7, 0.9, 0.5, 0.5],
    [0.59, 0.35, 0.99, 0.5, 0.6]
]

max_i = 0
max_j = 0
max_value = 0

M = len(scores)
N = len(scores[0])

for i in range(M):
    for j in range(N):
        current_value = scores[i][j]
        if current_value > max_value:
            max_value = current_value
            max_i = i
            max_j = j

print(max_value)
print(max_i)
print(max_j)

print('**********')
# Пользователь даёт нашей умной колонке Алевтина задание: поставить будильники особым образом.
# Будильники должны звонить каждые пять часов, начиная с 10 часов утра.
# Причём будильник должен прозвенеть дважды в час с перерывом в полчаса. Например, в 15:00 и 15:30.
#
# Необходимо вывести на экран время в которое прозвенят будильники.

hours = list(range(10, 24, 5))
mins = list(range(0, 60, 30))

for h in hours:
    for m in mins:
        print(f'{h:02}:{m:02}')

print('**********')
# Пусть пользователь задаёт будильники на каждые два часа, начиная с 9 часов утра.
# Причём будильники должны звонить четырежды в указанный час (интервал 15 минут),
# например в 11:00, 11:15, 11:30 и 11:45.
#
# Напишите вложенный цикл, который будет выводить на экран время, в которое прозвенят будильники.

hours = list(range(9, 24, 2))
mins = list(range(0, 60, 15))

for h in hours:
    for m in mins:
        print(f'{h:02}:{m:02}')

print('**********')
# Условие задачи. Дан список строк str_list = ['text', 'morning', 'notepad', 'television', 'ornament'].
# Необходимо подсчитать, сколько всего раз во всех строках списка встречается буква 'e'.

str_list = ['text', 'morning', 'notepad', 'television', 'ornament']

count_e = 0

for word in str_list:
    for letter in word:
        if letter.lower() == 'e':
            count_e += 1

print(f"Total e letters:{count_e}")

print('**********')
str_list = ['text', 'morning', 'notepad', 'television', 'ornament']  # заданный список строк
count = 0  # задаём начальное количество символов 'e'
# создаём цикл по элементам списка str_list
for text in str_list:
    # увеличиваем количество символов 'e'
    count += text.count('e')  # .count() считает, сколько раз символ встречается в строке text
print("Count symbol 'e':", count)  # выводим результат

print('**********')
list_to_string = " ".join(str_list)
count = list_to_string.count('e')
print("Count symbol 'e':", count)

print('**********')
# Дана двумерная матрица 3x3 (список списков). Необходимо определить максимумы и минимумы в каждой её строке.
random_matrix = [
    [9, 2, 1],
    [2, 5, 3],
    [4, 8, 5]
]

max_row = []
min_row = []

for row in random_matrix:
    max_value = row[0]
    min_value = row[0]
    for elem in row:
        if elem > max_value:
            max_value = elem
        if elem < min_value:
            min_value = elem
    max_row.append(max_value)
    min_row.append(min_value)

print(max_row)
print(min_row)

print('**********')
# Нам предоставлены данные о баллах пяти студентов по трём экзаменам.
# Данные находятся во вложенном списке student_scores.
# Каждая строка таблицы соответствует номеру студента,
# а каждый из трёх столбцов соответствует номеру экзамена (первый — математика, второй — информатика, третий — русский язык).
# Необходимо найти средний балл студентов по каждому из экзаменов и общий средний балл по всем экзаменам.
student_scores = [
    [56, 90, 80],
    [80, 86, 92],
    [91, 76, 89],
    [91, 42, 60],
    [65, 30, 90]
]

avarage_lst = []

sum_math = 0
sum_it = 0
sum_lang = 0
sum_total = 0
students = len(student_scores)
subjects = len(student_scores[0])

for student_num in range(students):
    for subject_num in range(subjects):
        subject = student_scores[student_num][subject_num]
        sum_total += subject

        if subject_num == 0:
            sum_math += subject
        elif subject_num == 1:
            sum_it += subject
        elif subject_num == 2:
            sum_lang += subject

print(sum_math / students)
print(sum_it / students)
print(sum_lang / students)
print(sum_total / (students * subjects))

table = [
    [1, 3, 6],
    [4, 6, 8],
    [10, 33, 53]
]

sum_first_column = 0
for row in table:
    sum_first_column += row[0]

print(sum_first_column)

# Напишите код, который определяет, является ли матрица квадратной (то есть количество строк равно количеству столбцов).
# В конце программа должна выводить на экран значение True или False в зависимости от заданной матрицы.
test_matrix1 = [
    [1, 2, 3],
    [7, -1, 2],
    [123, 2, -1]
]

test_matrix2 = [
    [1, 2, 3],
    [7, -1, 2],
    [123, 2, -1],
    [123, 5, 1]
]


def is_matrix_square(matrix):
    return len(matrix) == len(matrix[0])


print(is_matrix_square(test_matrix1))
print(is_matrix_square(test_matrix2))

# В нашем распоряжении всё та же информация о динамике пользователей user_dynamics в нашем приложении.
#
# Мы дали начинающему программисту задание написать программу, которая находит номер последнего дня,
# когда наблюдался отток клиентов (динамика была отрицательной),
# а также находит количество ушедших в этот день клиентов.
user_dynamics = [-5, 2, 4, 8, 12, -7, 5]

last_negative_day = 0
last_negative_value = 0

for index, value in enumerate(user_dynamics):
    if value < 0:
        last_negative_day = index + 1
        last_negative_value = value

print(last_negative_day)
print(last_negative_value)


def from_tree(num):
    result = False
    while True:
        if num % 3 == 0:
            num = num // 3

            if num == 1:
                result = True
                break
        else:
            break

    return result


print(from_tree(81))


print('*********')
# Допишите программу, которая проверяет гипотезу Сиракуз. Гипотеза Сиракуз заключается в том,
# что любое натуральное число можно свести к 1, если повторять над ним следующие действия:
#
# если число чётное, разделить его пополам, т. е. n = n // 2;
# если нечётное — умножить на 3, прибавить 1 и результат разделить на 2, т. е. n = (n * 3 + 1) // 2.


n = 19 #задаём число
#создаём бесконечный цикл
while True:
    if n % 2 == 0:
        n = n // 2
    else:
        n = (n * 3 + 1) // 2

    if n == 1: #если результат равен 1,
        print('Syracuse hypothesis holds') #выводим утвердительное сообщение
        break


print('*********')
# Дан словарь my_dict. Его значения могут быть как строками (тип str), так и числами (типы int и float).
# Посчитайте, сколько значений в словаре my_dict являются числами. Используйте в своём коде оператор continue.

my_dict = {'a': 15, 'b': 10.5, 'c': '15', 'd': 50, 'e': 15, 'f': '15'}

print()
count_any_numbers = 0

for i in my_dict:
    if isinstance(my_dict[i], str):
        continue

    count_any_numbers += 1

print(count_any_numbers)


print('*********')
# Подсчитать количество вхождений каждого символа в заданном тексте.
# В результате работы программы должен быть сформирован словарь,
# ключи которого — символы текста, а значения — количество вхождений символа в тексте.
text = """
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had 
not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her 
and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, 
but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with 
cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of 
the shelves as she passed; it was labelled `ORANGE MARMALADE', but to her great disappointment it was empty: 
she did not like to drop the jar for fear of killing somebody, 
so managed to put it into one of the cupboards as she fell past it.

`Well!' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling down stairs! 
How brave they'll all think me at home! Why, I wouldn't say anything about it, 
even if I fell off the top of the house!' (Which was very likely true.)
"""

text = text.lower()

text = text.replace(" ", '')
text = text.replace("\n", '')

print(text)

letters_dict = {}

for i in text:
    if i not in letters_dict:
        letters_dict[i] = 1
    else:
        letters_dict[i] += 1

print(letters_dict)


print('*********')
# Подсчитать количество вхождений каждого слова в заданном тексте.
# В результате работы программы должен быть сформирован словарь,
# ключи которого — слова текста, а значения — количество вхождений слов в тексте.
text = """
She sells sea shells on the sea shore;
The shells that she sells are sea shells I am sure.
So if she sells sea shells on the sea shore,
I am sure that the shells are sea shore shells.
"""

text = text.lower()
text = text.replace(';', " ")
text = text.replace('.', " ")
text = text.replace(',', " ")
text = text.replace('\n', " ")
# text = text.replace('  ', " ")

text = ' '.join(text.split())
lst = text.split()

word_dict = {}

for word in lst:
    if word not in word_dict:
        word_dict[word] = 1
    else:
        word_dict[word] += 1

print(word_dict)