import numpy as np

from optimisation.setup import Setup

class TestProblem1(Setup):
    """
    Constrained Himmelbleau
    Optimum at: x = (3.0, 2.0), f = 0.0
    Optimum at: x = (-2.805118, 3.131312), f = 0.0
    Optimum at: x = (-3.779310, -3.283186), f = 0.0
    Optimum at: x = (3.584428, -1.848126), f = 0.0
    Bounds: x1 = [-6, 6], x2 = [-6, 6]
    Constraints: g <= 0
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Himmelbleau'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0  # 1
        self.var_opt = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
        self.f_opt = np.array([[0.0], [0.0], [0.0], [0.0]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = -6.0
            self.ub[i] = 6.0

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        # cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        obj = np.zeros(self.n_objectives)
        obj[0] = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

        return obj

    def cons_func_specific(self, x):
        x1 = 7 * x[0] / 24 + 1.75
        x2 = x[1] / 3 + 2

        g = np.zeros(self.n_constraints+1)
        g[0] = x1*np.sin(4*x1) + 1.1*x2*np.sin(2*x2)
        g[1] = -x1 - x2 + 3
        cons = np.max(g)
        cons *= -1

        return np.array(cons)


class TestProblem2(Setup):
    """
    Rosenbrock with modified Coello constraint
    Optimum at: x = (1, 1), f = 0
    Bounds: x1 = [-1.5, 1.5], x2 = [-1, 2.5]
    Constraints: g <= 0
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Rosenbrock'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0  # 1

        self.var_opt = np.array([1.0*np.ones(self.dim)])
        self.f_opt = np.array([0.0])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = -1.5
            self.ub[i] = 2.5
        #self.lb[0] = -1.5
        #self.lb[1] = -1.0
        #self.ub[0] = 1.5
        #self.ub[1] = 2.5

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=-1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        # cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):

        obj = np.zeros(self.n_objectives)
        for i in range(len(x) - 1):
            obj[0] += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2

        return obj

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        g = np.zeros(self.n_constraints+1)
        g[0] = -x1**2 + (3*x2/4 - 1.2)**1 + 1 + 0.3*np.cos(12*np.arctan(x1/(x2/3 - 1.2)))
        g[1] = x1**2 + 0.5*(x2 - 1.5)**2 - 2
        cons = np.max(g)
        cons *= -1

        return np.array(cons)

class TestProblem3(Setup):
    """
    Modified Branin from Sasena, 2002
    Optimum at: x = (-pi, 12.275), f = 0.397887
    Optimum at: x = (pi, 2.275), f = 0.397887
    Optimum at: x = (9.42478, 2.475), f = 0.397887
    Bounds: x1 = [-5, 10], x2 = [0, 15]
    Constraints: g <= 0
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'scaled_newBranin'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0  # 1
        self.var_opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.f_opt = np.array([[-1.04739], [-1.04739], [-1.04739]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = -5.0
        self.lb[1] = 0.0
        self.ub[0] = 10.0
        self.ub[1] = 15.0

        self.a = 1
        self.b = 5.1/(4*np.pi**2)
        self.c = 5/np.pi
        self.d = 6
        self.h = 10
        self.ff = 1/(8*np.pi)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=[2.0 , 2.0], scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        # cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        obj = np.zeros(self.n_objectives)
        obj[0] = (1/51.95)*((self.a*x2 - self.b*x1**2 + self.c*x1 - self.d)**2 + self.h*(1-self.ff)*np.cos(x1) - 44.81)

        return obj

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        cons = np.zeros(self.n_constraints)
        cons[0] = (1/51.95)*((self.a*x2 - self.b*x1**2 + self.c*x1 - self.d)**2 + self.h*(1-self.ff)*np.cos(x1) - 5 + self.h)
        cons *= -1

        return cons

class TestProblem4(Setup):
    """
    Log-Goldstein-Price with Sasena constraint, 2002
    Optimum at: x = (-0.5, -0.5), f = -2.2
    Bounds: x1 = [-0.5, 1.5], x2 = [-0.5, 2.5]
    Constraints: g <= 0
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'log_Goldstein_Price'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0  #1
        self.var_opt = np.array([[-0.5, -0.5]])
        self.f_opt = np.array([[-2.2]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = -0.5
        self.lb[1] = -0.5
        self.ub[0] = 1.5
        self.ub[1] = 2.5

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=[0.5, 1.0], scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        # cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        obj = np.zeros(self.n_objectives)
        obj[0] = (1/2.427)*(np.log(
            (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))*(30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
        ) - 8.693)

        return obj

    def cons_func_specific(self, x):
        x1 = (x[1] + 0.45)/2
        x2 = (x[0] + 0.5)/0.65

        g = np.zeros(self.n_constraints+2)
        g[0] = (x1 - 1)**3 - x2 + 1
        g[1] = x1 + x2 - 2
        g[2] = -x1
        cons = np.max(g)
        cons *= -1

        return np.array(cons)

class TestProblem5(Setup):
    """
    Constrained Styblinski-Tang
    Optimum at: x = (-2.903534, ..., -2.903434), f = -39.16599d
    Bounds: x1, x2 = [-5, 5]
    Constraints: g <= 0
    """

    def __init__(self, dim=2):
        super().__init__()
        self.problem_name = 'Styblinski_Tang'
        self.dim = dim
        self.n_objectives = 1
        self.n_constraints = 0 # 1
        self.var_opt = np.array([-2.903534*np.ones(self.dim)])
        self.f_opt = np.array([[-39.16599*self.dim]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = -5
            self.ub[i] = 5

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        #cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):

        obj = np.zeros(self.n_objectives)
        for i in range(self.dim):
            obj[0] += (x[i]**4 - 16*x[i]**2 + 5*x[i])

        return 0.5*obj

    def cons_func_specific(self, x):
        x1 = (x[1] + 2)/2
        x2 = (x[0] + 4)/1.5

        g = np.zeros(self.n_constraints)
        g[0] = -x2*np.cos(x2 + x1 + x1**2) + x1**3
        g *= -1

        return g

class TestProblem6(Setup):
    """
    Unconstrained Mystery Function
    Optimum at: x = (2.5044, 2.5778), f = -1.4565
    Bounds: x1, x2 = [0, 5]
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Mystery_function'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0 # 1
        self.var_opt = np.array([[2.5044, 2.5778]])
        self.f_opt = np.array([[-1.4565]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 0.0
        self.lb[1] = 0.0
        self.ub[0] = 5.0
        self.ub[1] = 5.0

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=[1.0, 1.0], scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        #cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        obj = np.zeros(self.n_objectives)
        obj[0] = 2 + 0.01*(x2 - x1**2)**2 + (1 - x1)**2 + 2*(2-x2)**2 + 7*np.sin(0.5*x1)*np.sin(0.7*x1*x2)

        return obj

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        g = np.zeros(self.n_constraints)
        g[0] = 0
        g *= -1

        return g

class TestProblem7(Setup):
    """
    Unconstrained Michalewicz Function
    Optimum at: x = (2.20 , 1.57), f = -1.8013 (2D)
    Bounds: x_i = [0, pi]
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Michalewicz_function'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0 # 1
        self.var_opt = np.array([[2.20, 1.57]])
        self.f_opt = np.array([[-1.8013]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = 0.0
            self.ub[i] = np.pi
        self.m = 5

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        #cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):

        obj = np.zeros(self.n_objectives)
        for i in range(self.dim):
            obj[0] += np.sin(x[i])*(np.sin((i+1)*x[i]**2/np.pi))**(2*self.m)

        return -obj

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        g = np.zeros(self.n_constraints)
        g[0] = 0
        g *= -1

        return g

class TestProblem8(Setup):
    """
    Unconstrained, shifted Rastrigin Function
    Optimum at: x = (0.5, ..., 0.5), f = 0
    Bounds: x_i = [-5.12, 5.12]
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Rastrigin_function'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0 # 1
        self.var_opt = np.array([0.5*np.ones(self.dim)])
        self.f_opt = np.array([[0]])

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = -5.12
            self.ub[i] = 5.12

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        #cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x = x - 0.5

        obj = np.zeros(self.n_objectives)
        for i in range(self.dim):
            obj[0] += x[i]**2 - 10*np.cos(2*np.pi*x[i])

        return obj + 10*self.dim

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        g = np.zeros(self.n_constraints)
        g[0] = 0
        g *= -1

        return g

class TestProblem9(Setup):
    """
    Unconstrained, Fletcher-Powell Function
    Optimum at: x = alpha, f = f_bias
    Bounds: x_i = [-pi, pi]
    """

    def __init__(self):
        super().__init__()
        self.problem_name = 'Fletcher_Powell'
        self.dim = 2
        self.n_objectives = 1
        self.n_constraints = 0 # 1

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        for i in range(self.dim):
            self.lb[i] = -np.pi
            self.ub[i] = np.pi

        self.f_bias = -460
        # seeds = [0, 555, 17]
        # seeds = [10000, 0, 334]
        seeds = [25, 25, 0]
        self.random_state = np.random.RandomState(seed=seeds[0])  # good: 0, 3, 25 (very interesting), 555
        self.a = self.random_state.randint(-100, 100, size=(self.dim, self.dim))
        self.random_state = np.random.RandomState(seed=seeds[1])  # good: 0, 3, 25 (very interesting), 555
        self.b = self.random_state.randint(-100, 100, size=(self.dim, self.dim))
        self.random_state = np.random.RandomState(seed=seeds[2])  # good: 0, 3, 25 (very interesting), 555
        self.alpha = self.random_state.randint(-np.pi, np.pi, size=self.dim)
        print(f"Optimum: {self.alpha}")

        self.var_opt = np.array([self.alpha])
        self.f_opt = np.array([[self.f_bias]])

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c',
                           lower=self.lb, upper=self.ub,
                           value=1.0*np.ones(self.dim), scale=1.0, var_opt=self.var_opt, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        #cons = self.cons_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):

        obj = np.zeros(self.n_objectives)
        for i in range(self.dim):
            obj[0] += (self.A(i) - self.B(x, i))**2

        return obj + self.f_bias

    def cons_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        g = np.zeros(self.n_constraints)
        g[0] = 0
        g *= -1

        return g

    def B(self, x, i):
        B_coeff = 0
        for j in range(self.dim):
            B_coeff += self.a[i, j]*np.sin(x[j]) + self.b[i, j]*np.cos(x[j])

        return B_coeff

    def A(self, i):
        A_coeff = 0
        for j in range(self.dim):
            A_coeff += self.a[i,j]*np.sin(self.alpha[j]) + self.b[i,j]*np.cos(self.alpha[j])

        return A_coeff
