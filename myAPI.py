import numpy as np


def myLen(list) -> int:
    return len(list)


if __name__ == '__main__':
    list = [1,2,3]
    print(myLen(list))
    len(list)
    list.pop()
    print(type(list))


