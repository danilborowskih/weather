import  numpy as np
def composition(k: int, x_list, x, n) -> np.float64:
    """lagrange subfunction"""
    p = np.float64(1)

    for j in range( n ):
        if j==k:
            p *= 1
        else:
            p *= (x - x_list[j]) / (x_list[k] - x_list[j])
    return p

def Lagrange( x_list=[3,5,6,7], y_list=[5,8,10,12], x=4.3 ) ->np.float64:
    """"return Lagrange interpolation"""
    n = len(x_list)
    y = np.float64(0)
    for k in range(n):
        y += y_list[k]*composition(k, x_list, x, n)
    return y 

if __name__=="__main__":
    Lagrange()