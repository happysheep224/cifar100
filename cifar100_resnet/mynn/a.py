def A():
    print("Hello, I am A :{}".format(__name__))
A()

def C():
    print("Hello, I am C:{}".format(__name__))
C()
if __name__ == '__main__':
    print("A is runing")