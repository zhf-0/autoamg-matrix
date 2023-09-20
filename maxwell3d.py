#!/usr/bin/env python3
# 
import os
import numpy as np
import scipy.sparse as sparse

import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

# solver
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve, cg, norm
from scipy.sparse import csr_matrix, spdiags, eye, bmat, save_npz

from fealpy.functionspace import FirstNedelecFiniteElementSpace3d 
from fealpy.mesh import TetrahedronMesh
from fealpy.pde.MaxwellPDE_3d import MaxwellPDE 

import torch
import torch_geometric.data as pygdat
from torch_geometric.utils import degree


class Sindata(MaxwellPDE):
    def __init__(self, beta=1, k=1):
        C = CoordSys3D('C')
        #f = 1*C.i + sym.sin(sym.pi*C.x)*C.j + sym.sin(sym.pi*C.z)*C.k 
        #f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j + C.x*sym.sin(sym.pi*C.z)*C.k 
        #f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j + C.z*C.k 
        f = sym.sin(sym.pi*C.z)*C.i + sym.sin(sym.pi*C.x)*C.j + sym.sin(sym.pi*C.y)*C.k 
        #f = sym.sin(sym.pi*C.x)*C.i + sym.sin(sym.pi*C.y)*C.j + sym.sin(sym.pi*C.z)*C.k 
        #f = C.x**3*C.i + C.y**3*C.j + C.z**3*C.k
        #f = C.y*C.i + 2*C.x*C.j + C.z*C.k

        #f = (C.x**2-C.x)**2*(C.y**2-C.y)**2*(C.z**2-C.z)**2
        #f = f*C.i + sym.sin(C.x)*f*C.j + sym.sin(C.y)*f*C.k
        super().__init__(f, beta, k)

    def domain(self):
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

class CreateData:
    def __init__(self,num):
        self.num = num
        self.root_path = './Max3dData/'
        os.makedirs(self.root_path,exist_ok=True)

        np.random.seed(0)
        self.mesh = np.random.randint(10,40,num)
        self.Beta = np.random.uniform(1,15,num)
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.beta = 1.0

    def Process(self):
        print('begin to process')
        for i in range(self.num):
            self.nx = int(self.mesh[i])
            self.ny = self.nx
            self.nz = self.nx
            self.beta = self.Beta[i]


            print('========================================================')
            print(f'begin to run, mat id = {i}')
            print(f'nx = {self.nx}, ny = {self.ny}, nz = {self.nz},')
            print(f'wave number = {self.beta}')

            print('begin to generate matrix')
            A = self.GenerateMat()
            print(f'nrow = {A.shape[0]}, nnz = {A.nnz}')

            print('save matrix to file')
            A_path = self.root_path + f'csr_A{i}.npz'
            save_npz(A_path, A)

            print('begin to generate graph ')
            graph = self.CreateGraph(A)

            print('save graph to file')
            graph_path = self.root_path + f'graph{i}.npz'
            torch.save(graph,graph_path)
            
    def GenerateMat(self):
        pde = Sindata(beta=self.beta)
        domain = pde.domain()
        mesh = TetrahedronMesh.from_box(domain, nx=self.nx, ny=self.ny, nz=self.nz)
        space = FirstNedelecFiniteElementSpace3d(mesh)

        M = space.mass_matrix()
        A = space.curl_matrix()
        b = space.source_vector(pde.source)
        B = A - pde.beta * M 

        Eh = space.function()
        isDDof = space.set_dirichlet_bc(pde.dirichlet, Eh)
        b = b - B@Eh
        b[isDDof] = Eh[isDDof]

        bdIdx = np.zeros(B.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, B.shape[0], B.shape[0])
        T = spdiags(1-bdIdx, 0, B.shape[0], B.shape[0])
        B = T@B@T + Tbd

        eps = 10**(-15)
        B.data[np.abs(B.data) < eps ] = 0
        B.eliminate_zeros()
        return B



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
    mat = CreateData(200)
    mat.Process()


if __name__ == '__main__':
    main()
