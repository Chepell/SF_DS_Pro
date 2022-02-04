# age = int(input("How old are you?"))
#
# if age > 100 or age <= 0:
#     raise ValueError(
#         "You are too old or don't exist")  # Ты слишком стар либо не существуешь (Тебе не может быть столько лет)
#
# print(f"You are {age} years old!")  # Возраст выводится только в том случае, если пользователь ввёл правильный возраст.

try:
    age = int(input("How old are you?"))

    if age > 100 or age <= 0:
        raise ValueError("Too old")
except ValueError:
    print("Wrong age")
else:
    print(f"You are {age} years old!")  # Возраст выводится только в случае, если пользователь ввёл правильный возраст.