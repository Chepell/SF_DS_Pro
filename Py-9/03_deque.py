# Напишите функцию brackets(line), которая определяет, является ли последовательность из круглых скобок правильной.
from collections import deque


def brackets(text: str):
    dq = deque()
    for ch in text:
        if ch == '(':
            dq.append(ch)
        elif ch == ')':
            if len(dq) == 0:
                return False
            else:
                dq.pop()

    if len(dq) == 0:
        return True
    else:
        return False


print(brackets("vgbddfgbfdg(vfd(vfdsvsd)(vdfvbfg)bgfbgf)bgfbfgbgf"))
# True
print(brackets("mlkmvlfds"))
# True
print(brackets("43fbrbte(gteg6h(j867jgfer)g354j896(g546rwc)vt3bg65)h647hj8)j986k57"))
# False
