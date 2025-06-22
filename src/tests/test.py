class Foo:
    def __init__(self):
        self.attributes = {'1': 'a', '2': 'b', '3': 'c'}
        self.num = 3
        return None

    def __next__(self):
        return next(iter(self.attributes))
    def __iter__(self):
        return iter(self.attributes)


foo = Foo()

for f in foo:
    print(next(foo))
    print(f)

a = foo.num

print(a, foo.num)
a = 4
print(a, foo.num)
