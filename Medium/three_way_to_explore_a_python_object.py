
class Person:
    def __init__(self, first_name="", last_name=""):
        self.first_name=first_name
        self.last_name=last_name

Author=Person("Yang", "Zhou")
a="yang"
b=100

print(type(a))
print(type(b))

print(type(Author))

print(type(int))

class Developer(Person):
    pass

class TopDeveloper(Developer):
    pass

one_guy=Developer()
Yang=TopDeveloper("Yang", "Zhou")

isinstance(Yang, Person)
isinstance(one_guy, TopDeveloper)

type(Yang).mro() # reveal class hierachi

setattr(Yang, "first_name", "hello")
Yang.first_name