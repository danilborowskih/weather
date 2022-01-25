import numpy as np
from numpy.core.function_base import linspace
def Div_Diff(x_list, func,y_list):
    divided_differences = np.float64(0)
    for j in range(len(x_list)):
        p=np.float64(1)
        for i in range(len(x_list)):
            if i!=j:
                p*=(x_list[j] - x_list[i])

        if y_list==None:divided_differences += func(x_list[j])/p
        else: divided_differences += y_list[j]/p

    return divided_differences

def Newton(x_list , x_for_interpolation, func=lambda x: 1/(1+x**2), y_list=[]):
    y_interpolated_list = []
    for x in x_for_interpolation:
        y_interpolated = np.float64(0)
        
        for n in range(len(x_list)):
            diviaded_values = np.float64(1)
            for i in range(n):
                diviaded_values *= x - x_list[i]
            y_interpolated += diviaded_values*Div_Diff(x_list[0:n+1],func=func, y_list=y_list)
        y_interpolated_list += [y_interpolated]

    return y_interpolated_list

if __name__=="__main__":
    n=5
    #print(Newton(np.linspace(-5,5,n),[0.75, 1.75, 2.75, 3.75, 4.75]))
    x_list7=[2**i for i in range(0,7)]
    x_list8=[2**i for i in range(0,8)]
    func = lambda x: x**7+x**5+6*x+1
    print((Div_Diff(x_list7,func=func)+Div_Diff(x_list8,func=func))/2)