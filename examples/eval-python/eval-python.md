
~~~{.python}
for i in range(10):
    print(i, end = '')
~~~

~~~{.python .eval}
print("defining foo")
def foo():
    for i in range(10):
        print(-i, end = '')
foo()
~~~

~~~{.python .eval}
print("calling foo from different block")
foo()
~~~
