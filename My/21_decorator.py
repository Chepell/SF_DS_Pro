# Напишите функцию-декоратор-логгер logger(name).
#
# При создании декоратора передаётся имя логгера,
# которое выводится при каждом запуске декорируемой функции.
#
# Декорированная функция должна печатать:
#
# перед запуском основной:
# <имя логгера>: Function <имя декорируемой функции> started
# после запуска основной:
# <имя логгера>: Function <имя декорируемой функции> finished


def logger(name):
    def decorator(func):
        def decorated_func(*args, **kwargs):
            func_name = func.__name__
            text = f"{name}: Function {func_name} started"
            print(text)
            result = func(*args, **kwargs)
            text = f"{name}: Function {func_name} finished"
            print(text)
            return result

        return decorated_func

    return decorator


@logger('MainLogger')
def root(val, n=2):
    res = val ** (1 / n)
    return res


print(root(25))
