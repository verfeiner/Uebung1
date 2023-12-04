import random

import numpy as np
a=[]
for i in range(0,100):
    num=random.randint(0,100)
    a.append(num)

print(a)
m=0
n=len(a)
for num in a:
    if num>m:
       m=num
print(m)
random_numbers = [random.randint(0, 100) for _ in range(100)]

# 统计每个数字的出现次数
frequency = {}
for number in random_numbers:
    if number in frequency:
        frequency[number] += 1
    else:
        frequency[number] = 1

# 找到出现次数最多的数字和次数
most_common_number = max(frequency, key=frequency.get)
most_common_count = frequency[most_common_number]
print(most_common_count,most_common_number)

frequency = {1: 5, 2: 3, 3: 7, 4: 2}
print(frequency)