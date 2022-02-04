def mygen():
    i = 7
    print('hello')
    while i > 0:
        i -= 1
        yield i


mg = mygen()
print(next(mg))
print(next(mg))
print(next(mg))
print(next(mg))
print(next(mg))
print(next(mg))
print(next(mg))


# Напишите бесконечный итератор по списку.
#
# Для этого создайте генератор с названием inf_iter,
# который принимает на вход список и возвращает элементы из него через yield.
#
# Когда элементы в списке заканчиваются,
# генератор снова возвращает элементы из списка, начиная с нулевого.

def inf_iter(l_in):
    while True:
        for i in l_in:
            yield i


l = [101, 102, 103]
gen = inf_iter(l)
for _ in range(10):
    print(next(gen))


# Напишите генератор group_gen(n).
# Он должен при каждом вызове выдавать порядковый номер от 1 до n (включая n).
# После достижения n генератор должен снова возвращать номера, начиная с 1.

def group_gen(n):
    while True:
        for i in range(1, n + 1):
            yield i


groups = group_gen(3)
for _ in range(10):
    print(next(groups))
