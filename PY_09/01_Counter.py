from collections import Counter

c = Counter()

c['red'] += 1
c['red'] += 10
print(c)

cars = ['red', 'blue', 'black', 'black', 'black', 'red', 'blue', 'red', 'white']

c = Counter(cars)
print(c)

print(c['black'])

print(c.values())
print(sum(c.values()))

c.most_common()