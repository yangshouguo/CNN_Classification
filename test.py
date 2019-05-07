from functools import  reduce

filter_sizes = [1,2,3,4]
print(reduce(lambda x,y:x+y, filter_sizes))