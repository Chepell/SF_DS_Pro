list1 = [10, 15]
tpl1 = tuple(list1)
dict1 = dict()
dict1[tpl1] = 'hello'
print(dict1)

dict1 = {'name': 'unknown'}
dict2 = {'name': 'Tom'}
print(dict1)
print(dict2)
print(dict1.keys())
print(dict1.values())

s1 = set('hello')
s2 = set(['w', 'o', 'w'])
print(s1.union(s2))
print(s1.intersection(s2))
print(s1.difference(s2))

result = 'age is ' + str(15)
print(result)

print("Повесть 'Капитанская дочка'")