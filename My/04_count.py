numbers = [7, 5, 1, 3, 1, 2, 5, 4, 9, 5]


cnt = numbers.count(5)
print(cnt)


my_books = ['book1', 'book2', 'book3', 'book4', 'book5']
tom_books = my_books.copy()
print(my_books)
print(tom_books)


my_orders = ['order1', 'order2', 'order3', 'order4', 'order5']
anne_order = my_orders[:]

print(my_orders)
print(anne_order)


all_things = ['order1', 'order2', 'order3']
only_books = ['book1', 'book2']

all_things.extend(only_books)

print(all_things)

nums = list(range(1, 11))
nums.reverse()
print(nums)


random_values = [3, 5, 0, -1, 2, 10, 15, -5]
random_values.sort()
print(random_values)

list1 = [5, 0.2, 'hello there', [1, 2, 3, 4], 'bye']
print(list1)


tpl = (255,)
print(tpl)

tpl3 = (15, 22, 0)
print(tpl3)