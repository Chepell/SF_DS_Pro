# Представьте себя администратором тайного интернет-сообщества «Три Слепые Мыши» с ограниченным членством.
# У каждого участника сообщества есть свой позывной и пароль.
# Позывные и пароли представлены в словаре secret_passwords.
# Ключи данного словаря — позывные, а значения — пароли.

secret_passwords = {
    'Enot': 'ulybaka',
    'Agent12': '1password1',
    'MouseLulu': 'myshkanaruhka'
}

while True:
    login_input = input('Print your name --> ')

    if login_input in secret_passwords:
        pass_input = input('Print your password --> ')

        if pass_input == secret_passwords[login_input]:
            print(f"Welcome {login_input}")
            break
        else:
            print(f"Dear {login_input}! You typed wrong password!")
    else:
        print(f"User with name: {login_input} doesn't exist!")

print('Inside!')
