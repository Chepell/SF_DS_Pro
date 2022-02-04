s1 = set()
print(s1)

s2 = {5, 10, 3, 2, 11}
print(s2)

s3 = set('wow thats cool')
print(s3)

s1 = set('abcde')
s1.add('hello')
print(s1)

s1 = set('abcde')
s1.remove('d')
print(s1)

s1 = set('abcdef')
s1.discard('f')
print(s1)

alpha_set = set('abcde')
name = set('bad boy')

new_set = alpha_set.union(name)
print(new_set)

num_set = set(range(0, 11))
date_num = set([1, 9, 4, 8])
print(num_set.union(date_num))


alpha_set = set('abcde')
name = set('bad boy')
print(alpha_set.intersection(name))


num_set = set(range(0, 11))
date_num = set([1, 9, 4, 8])
print(num_set.intersection(date_num))


alpha_set = set('abcde')
name = set('bad boy')
print(alpha_set.difference(name))
print(name.difference(alpha_set))


num_set = set(range(0, 11))
date_num = set([1, 9, 4, 8])
print(num_set.difference(date_num))