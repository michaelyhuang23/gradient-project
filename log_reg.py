from variable import Variable
import numpy as np
import math
class LogisticRegression:
    def __init__(self, var_len) -> None:
        '''
        var_len is the number of parameters that the model takes as input variables
        '''
        self.vars = [Variable(name=i) for i in range(1,var_len+1)]
        self.var_len = var_len
        self.coeffs = [Variable(name=-i) for i in range(1,var_len+1)]
        self.coeffs_val = {-i : np.random.uniform(-1, 1) for i in range(1,var_len+1)}
        self.linear = sum(coeff*var for coeff, var in zip(self.coeffs, self.vars))
        self.pred = 1/(1+math.e**(0-self.linear))

    def fit(self, Xs, ys, lr=0.01):
        '''
        Perform gradient descent on Xs, and ys with a set lr
        Xs is an iterable of input variables
        ys is an iterable of outputs
        '''
        loss_val = []
        for X, y in zip(Xs, ys):
            vars_val = {(i+1) : var for i, var in enumerate(X)}
            all_coeffs = vars_val
            all_coeffs.update(self.coeffs_val)
            loss = 0-y*Variable.log(self.pred) - (1-y)*Variable.log(1-self.pred) 
            loss_val.append(loss.eval(all_coeffs))
            loss_grad = loss.gradient(all_coeffs)
            for i, (coeff_name, val) in enumerate(all_coeffs.items()):
                if coeff_name > 0:
                    continue
                self.coeffs_val[coeff_name] -= loss_grad[i]*lr
        return sum(loss_val)/len(loss_val)

    def predict(self, Xs):
        '''
        Xs is an iterable of iterables
        each X in Xs is a set of input parameters
        The function returns a list of outputs, each output obtained independently from an X
        '''
        vals = []
        for X in Xs:
            vars_val = {(i+1) : var for i, var in enumerate(X)}
            all_coeffs = vars_val
            all_coeffs.update(self.coeffs_val)
            vals.append(self.pred.eval(all_coeffs))
        return vals

