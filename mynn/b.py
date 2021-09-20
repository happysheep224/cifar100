#import a
from a import A
def B():
    print("Hello, I am block B:{}".format(__name__))
B()
if __name__ == '__main__':
   print("B is runing")