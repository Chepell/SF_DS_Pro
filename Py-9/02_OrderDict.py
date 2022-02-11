data = [('Ivan', 19), ('Mark', 25), ('Andrey', 23), ('Maria', 20)]
client_ages = dict(sorted(data, key=lambda x: x[1]))
print(client_ages)

temps = [('2000', -4.4), ('2001', -2.5), ('2002', -4.4), ('2003', -9.5),
         ('2004', -8.2), ('2005', -1.6), ('2006', -5.9), ('2007', -2.4),
         ('2008', -1.7), ('2009', -3.5), ('2010', -12.1), ('2011', -5.8),
         ('2012', -4.9), ('2013', -6.1), ('2014', -6.9), ('2015', -2.7),
         ('2016', -11.2), ('2017', -3.9), ('2018', -2.9), ('2019', -6.5),
         ('2020', 1.5)]

# Напечатайте словарь из температур, отсортированный по уменьшению температуры
from collections import OrderedDict

sort_tem = OrderedDict(sorted(temps, key=lambda x: x[1], reverse=True))
print(sort_tem)
print('**********')
# Дан список кортежей ratings с рейтингами кафе. Кортеж состоит из названия и рейтинга кафе.
#
# Необходимо отсортировать список кортежей по убыванию рейтинга. Если рейтинги совпадают,
# то отсортировать кафе дополнительно по названию в алфавитном порядке.
ratings = [('Old York', 3.3), ('New Age', 4.6), ('Old Gold', 3.3), ('General Foods', 4.8),
           ('Belissimo', 4.5), ('CakeAndCoffee', 4.2), ('CakeOClock', 4.2), ('CakeTime', 4.1),
           ('WokToWork', 4.9), ('WokAndRice', 4.9), ('Old Wine Cellar', 3.3), ('Nice Cakes', 3.9)]

# Сначала сортировка по рейтигу по убыванию, для этого -, а затем сортировка по названию
sort_ratings = sorted(ratings, key=lambda x: (-x[1], x[0]))
sort_dict = dict(sort_ratings)
print(sort_dict)

rating2 = [('WokAndRice', 4.9), ('WokToWork', 4.9), ('Old York', 3.3), ('New Age', 4.6), ('Old Gold', 3.3),
           ('General Foods', 4.8), ('New Age', 4.6), ('Belissimo', 4.5), ('CakeAndCoffee', 4.2),
           ('CakeOClock', 4.2), ('CakeTime', 4.1), ('Nice Cakes', 3.9), ('Old Gold', 3.3), ('WokToWork', 4.9),
           ('WokAndRice', 4.9), ('Old Wine Cellar', 3.3), ('Old York', 3.3), ('Nice Cakes', 3.9)]
sort_ratings = sorted(rating2, key=lambda x: (-x[1], x[0]))
sort_dict = dict(sort_ratings)
print(sort_dict)