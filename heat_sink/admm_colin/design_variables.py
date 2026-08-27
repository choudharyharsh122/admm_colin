import numpy as np
import math

class DesignVariables:
    """
    Holds numpy arrays for the ADMM design variables a, b, ?.
    """
    def __init__(self, seed, size, Vmax):
        np.random.seed(seed)
        # Run this for Triangular elements
        # self.a   = np.random.randint(0, 2, size=(size, size)).astype(float)   # binary values {0,1}
        # self.b   = np.random.rand(size,size)                               # continuous [0,1]
        # self.lam = np.random.randn(size,size)                              # real numbers

        # self.a   = np.zeros((size, size))   # binary values {0,1}
        # self.b   = 0.4*np.ones((size, size))                               # continuous [0,1]
        # self.lam = np.zeros((size, size))                              # real numbers
        #self.rho = 0.5
        self.a   = np.zeros(2*size*size)   # binary values {0,1}
        self.b   = 0.4*np.ones(2*size*size)
        self.lam = 0*np.ones(2*size*size)                              # real numbers
        #self.b = np.random.rand(2*size*size)    
        #self.lam = np.random.rand(2*size*size)                             # continuous [0,1]
        #self.lam = 0.4*np.ones(2*size*size)
        #self.lam = 0.4*np.ones(2*size*size)

        # Run this for Square elemets
        # self.a   = np.random.randint(0, 2, size=size).astype(float)   # binary values {0,1}
        # self.b   = np.random.rand(size)                               # continuous [0,1]
        # self.lam = np.random.randn(size)                              # real numbers

        # self.a   = np.zeros(size)   # binary values {0,1}
        # self.b   = np.zeros(size)                               # continuous [0,1]
        # self.lam = np.zeros(size)                              # real numbers


    def set_a(self, prev_a):
        a1 = prev_a.copy() 
        a2 = self.a.copy()

        print("len of array a1 : ", len(a1))
        print("len of array a2 : ", len(a2))

        nx = int(2*math.sqrt(len(a1)/2))
        Nx = int(2*math.sqrt(len(a2)/2))

        print(f"nx : {nx}")
        print(f"Nx : {Nx}")

        for k in range(0, len(a1) - 1, 2):
    
            ###Setting value for the lower triangle cells###
            a2[(2*(k //nx)*nx) + 2*k ] = a1[k]
            a2[(2*(k //nx)*nx)  + 2*k + 2] = a1[k]
            a2[(2*(k //nx)*nx)  + 2*k + 3] = a1[k]
            a2[(2*(k //nx)*nx) + Nx  + 2*k + 2] = a1[k]
            
            ###Setting value for the upper triangle cells###
            a2[(2*(k //nx)*nx) + (2*k) +1] = a1[k+1]
            a2[(2*(k //nx)*nx) + (2*k) + Nx] = a1[k+1]
            a2[(2*(k //nx)*nx) + (2*k) + Nx + 1] = a1[k+1]
            a2[(2*(k //nx)*nx) + (2*k) + Nx + 3] = a1[k+1]
        
        self.a = a2

    def set_b(self, prev_b):
        b1 = prev_b.copy() 
        b2 = self.b.copy()

        # print("len of array a1 : ", len(b1))
        # print("len of array a2 : ", len(b2))

        nx = int(2*math.sqrt(len(b1)/2))
        Nx = int(2*math.sqrt(len(b2)/2))

        # print(f"nx : {nx}")
        # print(f"nx : {Nx}")

        for k in range(0, len(b1) - 1, 2):
    
            ###Setting value for the lower triangle cells###
            b2[(2*(k //nx)*nx) + 2*k ] = b1[k]
            b2[(2*(k //nx)*nx)  + 2*k + 2] = b1[k]
            b2[(2*(k //nx)*nx)  + 2*k + 3] = b1[k]
            b2[(2*(k //nx)*nx) + Nx  + 2*k + 2] = b1[k]
            
            ###Setting value for the upper triangle cells###
            b2[(2*(k //nx)*nx) + (2*k) +1] = b1[k+1]
            b2[(2*(k //nx)*nx) + (2*k) + Nx] = b1[k+1]
            b2[(2*(k //nx)*nx) + (2*k) + Nx + 1] = b1[k+1]
            b2[(2*(k //nx)*nx) + (2*k) + Nx + 3] = b1[k+1]
        
        self.b = b2

    def set_lambda(self, prev_lam):
        lam1 = prev_lam.copy() 
        lam2 = self.lam.copy()

        # print("len of array a1 : ", len(lam1))
        # print("len of array a2 : ", len(lam2))

        nx = int(2*math.sqrt(len(lam1)/2))
        Nx = int(2*math.sqrt(len(lam2)/2))

        # print(f"nx : {nx}")
        # print(f"nx : {Nx}")

        for k in range(0, len(lam1) - 1, 2):
    
            ###Setting value for the lower triangle cells###
            lam2[(2*(k //nx)*nx) + 2*k ] = lam1[k]
            lam2[(2*(k //nx)*nx)  + 2*k + 2] = lam1[k]
            lam2[(2*(k //nx)*nx)  + 2*k + 3] = lam1[k]
            lam2[(2*(k //nx)*nx) + Nx  + 2*k + 2] = lam1[k]
            
            ###Setting value for the upper triangle cells###
            lam2[(2*(k //nx)*nx) + (2*k) +1] = lam1[k+1]
            lam2[(2*(k //nx)*nx) + (2*k) + Nx] = lam1[k+1]
            lam2[(2*(k //nx)*nx) + (2*k) + Nx + 1] = lam1[k+1]
            lam2[(2*(k //nx)*nx) + (2*k) + Nx + 3] = lam1[k+1]
        
        self.lam = lam2

    def set_lam_const(self, const, size):

        self.lam = const*np.ones(2*size*size)
