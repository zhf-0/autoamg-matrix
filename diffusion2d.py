import os
import numpy as np
# from scipy.sparse.linalg import spsolve, inv, norm
import scipy.sparse as sparse

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

import torch
import torch_geometric.data as pygdat
from torch_geometric.utils import degree

def COO2CSR0(row,col,nnz,coo_i,coo_j,coo_val):
    '''
    coo to csr 
    '''
    csr_i = np.zeros(row+1,dtype=int)
    csr_j = np.zeros(nnz,dtype=int)
    csr_val = np.zeros(nnz,dtype=np.float64)

    for i in range(nnz):
        csr_i[coo_i[i] + 1] += 1

    num_per_row = csr_i.copy()

    for i in range(2,row + 1):
        csr_i[i] = csr_i[i] + csr_i[i-1]

    for i in range(nnz):
        row_idx = coo_i[i]
        begin_idx = csr_i[row_idx]
        end_idx = csr_i[row_idx+1]
        offset = end_idx - begin_idx - num_per_row[row_idx+1]
        num_per_row[row_idx + 1] -= 1

        csr_j[begin_idx + offset] = coo_j[i]
        csr_val[begin_idx + offset] = coo_val[i]

    return csr_i, csr_j, csr_val

class PDE:
    def __init__(self,x0,x1,y0,y1,blockx,blocky):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 
        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)

        shape = p.shape+(2,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx]
        # val[...,0,0] = 10.0
        val[...,0,1] = 1.0
        val[...,1,0] = 1.0
        val[...,1,1] = self.coef2[xidx,yidx]
        # val[...,1,1] = 2.0
        return val

    @cartesian
    def convection_coefficient(self, p):
        return np.array([0.0, 0.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 0*x + 0*y

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)



class CreateData:
    def __init__(self,num):
        self.num = num
        self.root_path = './Diff2dData/'
        os.makedirs(self.root_path,exist_ok=True)

        np.random.seed(0)
        self.mesh = np.random.randint(50,100,num)
        self.block = np.random.randint(10,20,num)
        self.nx = 0
        self.ny = 0
        self.blockx = 0
        self.blocky = 0

    def Process(self):
        print('begin to process')
        for i in range(self.num):
            self.nx = int(self.mesh[i])
            self.ny = self.nx
            self.blockx = int(self.block[i])
            self.blocky = self.blockx

            print('========================================================')
            print(f'begin to run, mat id = {i}')
            print(f'nx = {self.nx}, ny = {self.ny}')
            print(f'blockx = {self.blockx}, blocky = {self.blocky}')

            print('begin to generate matrix')
            A = self.GenerateMat()
            print(f'nrow = {A.shape[0]}, nnz = {A.nnz}')

            print('save matrix to file')
            A_path = self.root_path + f'csr_A{i}.npz'
            sparse.save_npz(A_path, A)

            print('begin to generate graph ')
            graph = self.CreateGraph(A)

            print('save graph to file')
            graph_path = self.root_path + f'graph{i}.npz'
            torch.save(graph,graph_path)
            

    def GenerateMat(self):
        pde = PDE(0,1,0,1,self.blockx,self.blocky)
        domain = pde.domain()
        mesh = MF.boxmesh2d(domain, nx=self.nx, ny=self.ny, meshtype='quad',p=1)

        # space = LagrangeFiniteElementSpace(mesh, p=1)
        space = ParametricLagrangeFiniteElementSpace(mesh, p=1)
        uh = space.function() 	
        A = space.stiff_matrix(c=pde.diffusion_coefficient)
        # B = space.convection_matrix(c=pde.convection_coefficient)
        # M = space.mass_matrix(c=pde.reaction_coefficient)
        F = space.source_vector(pde.source)
        # A += B 
        # A += M
        
        bc = DirichletBC(space, pde.dirichlet)
        A, F = bc.apply(A, F, uh)

        eps = 10**(-15)
        A.data[np.abs(A.data) < eps ] = 0
        A.eliminate_zeros()
        return A


    def CreateGraph(self,A):
        row, col = A.nonzero()
        edge_weight = torch.zeros(A.nnz,dtype=torch.float32)
        for i in range(A.nnz):
            edge_weight[i] = abs(A[row[i],col[i]])

        row = torch.from_numpy(row.astype(np.int64))
        col = torch.from_numpy(col.astype(np.int64))
        x = degree(col,A.shape[0],dtype=torch.float32).unsqueeze(1)
        edge_index = torch.stack((row,col),0)

        graph = pygdat.Data(x=x,edge_index = edge_index,edge_weight = edge_weight)

        return graph



def main():
    mat = CreateData(100)
    mat.Process()


if __name__ == '__main__':
    main()
