dic1 = {'a': 1, 'b': 2}
dic2 = {'c': 3, 'd': 4}
dic3 = {**dic1, **dic2}

# search by key
print('a' in dic1)  # Return True

dic1.pop('a')  # remove
# del dic1['a'] #remove

# comprehension
a=range(1,11)
al=[i for i in a]
dic_comp={i:i**2 for i in range(1,11)}

#remove None
dic={'a':1, 'b':2, 'c':3, 'd':5, 'e':None}
remove_none={k:v for (k, v) in dic.items() if v is not None}

#use get mothod instead of []
print(dic.get('d'))
print(dic.get('f'))
# print(dic['f'])  #==> it raise KeyError

#filter by dictionary value using comprehension
dic={'John':150, "Felix":170, "Paul":145}
filtered_dict={k:v for (k,v) in dic.items() if v > 150}
