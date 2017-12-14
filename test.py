class A(object):
    va = 10

    def foo(self):
        self.va = 10
        print(A.va)
        print(self.va)

        self.va = 40
        print(A.va)
        print(self.va)

        va = 20
        print(va)

        A.va = 15
        print(A.va)
        print(self.va)


        # Script starts from here


obj1 = A()
obj2 = A()
obj1.foo()
obj2.foo()
print(A.va)

print(obj1.va)
print(obj2.va)
