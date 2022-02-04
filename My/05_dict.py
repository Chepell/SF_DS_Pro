empty_dict = dict()
print(empty_dict)

alphabet_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
print(alphabet_dict['d'])

place_and_money = {1: 100, 2: 50, 3: 10}
print(place_and_money[2])

place_and_money = {1: 100, 2: 50, 3: 10}
place_and_money[4] = 5

print(place_and_money)

place_and_money = {1: 100, 2: 50, 3: 10}
place_and_money[3] = 25

print(place_and_money)

place_and_money = {1: 100, 2: 50, 3: 10}
place_and_money.clear()

print(place_and_money)

place_and_money = {1: 100, 2: 50, 3: 10}

print(place_and_money.keys())

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
print(name_to_age.keys())

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
print(name_to_age.values())

place_and_money = {1: 100, 2: 50, 3: 10}
print(place_and_money.values())

place_and_money = {1: 100, 2: 50, 3: 10}
print(place_and_money.get(20, 0))

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
print(name_to_age.get('Denny', -1))

place_and_money = {1: 100, 2: 50, 3: 10}
place_and_money.update({4: 5, 5: 1})
print(place_and_money)

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
name_to_age.update({'Anne': 23, 'Phillip': 29})
print(name_to_age)

place_and_money = {1: 100, 2: 50, 3: 10}
print(place_and_money.pop(3))

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
print(name_to_age.pop('Anton'))

place_and_money = {1: 100, 2: 50, 3: 10}
place_and_money.setdefault(10, 1)
print(place_and_money)

name_to_age = {'Anne': 22, 'Anton': 27, 'Phillip': 30}
name_to_age.setdefault('Anne', 32)
print(name_to_age)

test_dict = {}
test_dict[5] = [3, 4, 5]
test_dict[(3, 4, 5)] = 'strong man'
print(test_dict)

test_dict2 = {}
test_dict2['name'] = 'Sancho'
test_dict2['surname'] = 'Panso'
test_dict2['info'] = {'age': 35, 'country': 'Mexico'}
print(test_dict2)

test_dict3 = dict()
test_dict3['info'] = [10, 15, 27]
test_dict3['about'] = {'game': 'football', 'period': 5}
test_dict3['about'] = 'dont know'
print(test_dict3)

