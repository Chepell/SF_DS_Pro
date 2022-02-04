# Запишите вместо вопросительных знаков выражение, которое вернет True,
# если указывается високосный год, иначе False.
# Например, x = 2000 -> True; x = 1900 -> False; и т.д.
# Если есть сомнения в том, какие именно годы високосные,
# обратитесь к Википедии:
# https://ru.wikipedia.org/wiki/Високосный_год#Григорианский_календарь

def is_leap_year(x):
    if x % 400 == 0:
        return True
    elif x % 100 == 0:
        return False
    elif x % 4 == 0:
        return True
    return False


print(is_leap_year(2300))


x = 10
y = 100
z = (x % 2 == 0) and (y >= 100)
print(z)