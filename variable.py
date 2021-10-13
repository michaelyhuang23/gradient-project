import numpy as np
import math

class Variable():
    infsmall = 1e-9
    def identity(values, name):
        '''
        returns the identity array: only the entry corresponding to name is 1
        '''
        val = np.zeros((len(values)))
        # the assumption of this code is that dict is ordered (which requires python>=3.7)
        idx = [i for i,key in enumerate(values.keys()) if key==name]
        if len(idx) == 0:
            raise ValueError(f'Cannot find key {name} in the input values')
        val[idx[0]] = 1
        return val

    def __init__(self, name=None, eval=None, gradient=None, tostring = None) :
        if name != None:
            self.name = name # its key in the evaluation dictionary
            self.tostring = lambda : name
        if tostring == None:
            self.tostring = lambda : name
        else:
            self.tostring = tostring
        if eval == None: # independent var
            self.eval = lambda values: values[self.name]
        else:
            self.eval = eval
        if gradient == None:
            self.gradient = lambda values: Variable.identity(values, name)
        else:
            self.gradient = gradient
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(
                eval = lambda values: self.eval(values) + other,
                gradient = self.gradient,
                tostring = lambda : f'({self.tostring()} + {other})'
            )
        return Variable(
            eval = lambda values: self.eval(values) + other.eval(values),
            gradient = lambda values: self.gradient(values) + other.gradient(values),
            tostring = lambda : f'({self.tostring()} + {other.tostring()})'
        )
    
    def __radd__(self, other):
        # other + self = self + other
        newVar = self.__add__(other)
        newVar.tostring = lambda : f'({other} + {self.tostring()})'
        return newVar
    
    def __sub__(self, other):
        # self - other  = self + (-other)
        newVar = self.__add__(other.__mul__(-1))
        if isinstance(other, (int, float)): newVar.tostring = lambda : f'({self.tostring()} - {other})'
        else: newVar.tostring = lambda : f'({self.tostring()} - {other.tostring()})'
        return newVar
    
    def __rsub__(self, other): 
        # other - self  = -(self - other)
        newVar = self.__sub__(other).__mul__(-1)
        newVar.tostring = lambda : f'({other} - {self.tostring()})'
        return newVar
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(
                eval = lambda values: self.eval(values) * other,
                gradient = lambda values: self.gradient(values) * other,
                tostring = lambda : f'({self.tostring()} * {other})'
            )
        return Variable(
            eval = lambda values: self.eval(values) * other.eval(values),
            gradient = lambda values: self.gradient(values) * other.eval(values) + other.gradient(values) * self.eval(values),
            tostring = lambda : f'({self.tostring()} * {other.tostring()})'
        )

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Variable(
                eval = lambda values: self.eval(values) ** other,
                gradient = lambda values: other*(self.eval(values) ** (other-1)) * self.gradient(values),
                tostring = lambda : f'({self.tostring()} ^ {other})'
            )
        return Variable(
            eval = lambda values: self.eval(values) ** other.eval(values),
            gradient = lambda values: other.eval(values)*self.eval(values)**(other.eval(values)-1)*self.gradient(values) + (self.eval(values)**other.eval(values))*np.log(self.eval(values))*other.gradient(values),
            tostring = lambda : f'({self.tostring()} ^ {other.tostring()})'
        )

    def __rpow__(self, other):
        # other is assumed to be constant (else __pow__ would be called)
        return Variable(
            eval = lambda values: other ** self.eval(values),
            gradient = lambda values: np.log(other) * other ** self.eval(values) * self.gradient(values),
            tostring = lambda : f'({other} ^ {self.tostring()})'
        )
    
    def __rmul__(self, other):
        # other * self = self * other
        newVar = self.__mul__(other)
        newVar.tostring = lambda : f'({other} * {self.tostring()})'
        return newVar

    def __truediv__(self, other):
        # self / other = self * other^(-1)
        newVar = self.__mul__(other.__pow__(-1))
        if isinstance(other, (int, float)): newVar.tostring = lambda : f'({self.tostring()} / {other})'
        else: newVar.tostring = lambda : f'({self.tostring()} / {other.tostring()})'
        return newVar
    
    def __rtruediv__(self, other):
        # other / self = (self / other)^(-1)
        newVar = self.__truediv__(other).__pow__(-1) 
        newVar.tostring = lambda : f'({other} / {self.tostring()})'
        return newVar
    
    def __call__(self, **kwargs):
        return self.eval(kwargs)
    
    def __str__(self):
        return self.tostring()
    
    def __repr__(self):
        return str(self)
    
    def grad(self, **kwargs):
        return self.gradient(kwargs)

    def log(var):
        if isinstance(var, (int, float)):
            return np.log(var)
        return Variable(
            eval = lambda values : np.log(var.eval(values)),
            gradient = lambda values : 1/var.eval(values) * var.gradient(values),
            tostring = lambda : f'ln({var})'
        ) 
    
    def exp(var):
        return math.e ** var
    
    def sin(var):
        if isinstance(var, (int, float)):
            return np.sin(var)
        return Variable(
            eval = lambda values : np.sin(var.eval(values)),
            gradient = lambda values : np.cos(var.eval(values)) * var.gradient(values),
            tostring = lambda : f'sin({var})'
        ) 

    def cos(var):
        if isinstance(var, (int, float)):
            return np.cos(var)
        return Variable(
            eval = lambda values : np.cos(var.eval(values)),
            gradient = lambda values : -np.sin(var.eval(values)) * var.gradient(values),
            tostring = lambda : f'cos({var})'
        )
    
    def tan(var):
        newVar = Variable.sin(var) / Variable.cos(var)
        newVar.tostring = lambda : f'tan({var})'
