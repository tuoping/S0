#!/usr/bin/python

import numpy as np
import glob,sys
from math import pi
import time
from MDAnalysis.coordinates.XTC import XTCReader
from copy import deepcopy

atom_types = None
def read_atom_types(filename="conf.lmp"):
    fp_head = open(filename,"r")
    fp_head.readline()
    natom = int(fp_head.readline().split()[0])
    print("natom = ", natom)
    for i in range(8):
        fp_head.readline()
    global atom_types
    atom_types = []
    for i in range(natom):
        type = fp_head.readline().split()[1]
        atom_types.append(type)


def read_xtc(traj):
    ts = traj._read_next_timestep()
    _cell = ts._unitcell
    cell = np.zeros(3,float)
    cell[0] = _cell[0]
    cell[1] = _cell[1]
    cell[2] = _cell[2]
    # if atom_types is None:
    #     read_atom_types()
    names = deepcopy(atom_types)
    q = ts.positions
    sq = np.zeros((len(ts.positions),3),float)
    for i in range(len(ts.positions)):
        sq[i,0] = q[i,0]/cell[0]
        sq[i,1] = q[i,1]/cell[1]
        sq[i,2] = q[i,2]/cell[2]
    return [cell, names, sq]

def read_lammpstrj(filedesc):
    # three comment lines
    for i in range(3): comment = filedesc.readline()
    # number of atoms
    natoms = int(filedesc.readline())
    print("natom = ", natoms)
    # 1 comment line
    comment = filedesc.readline()
    # assume orthorombic cell
    cell = np.zeros(3,float)
    for i in range(3): 
        # [cellmin, cellmax] = filedesc.readline().split()
        linec = filedesc.readline().split()
        cellmin = linec[0]
        cellmax = linec[1]
        cell[i] = float(cellmax) - float(cellmin)
    print(cell)
    # 1 comment line
    comment = filedesc.readline()
    names = np.zeros(natoms,'U2')
    q = np.zeros((natoms,3),float)
    sq = np.zeros((natoms,3),float)

    for i in range(natoms):
        line = filedesc.readline().split();
        names[i] = line[1] # atom type
        q[i] = line[2:5] # wrapped atomic coordinates
        sq[i,0] = float(q[i,0])/cell[0] # scaled atomic coordinates
        sq[i,1] = float(q[i,1])/cell[1] # scaled atomic coordinates
        sq[i,2] = float(q[i,2])/cell[2] # scaled atomic coordinates
    return [cell, names, sq]

def Sk(names, q, kgrid, e_A, e_B):
    # This is the un-normalized FT for the density fluctuations
    q_A = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_A ])
    n_A = len(q_A)
    print("Number of element A: ", n_A)
    if n_A > 0:
        FTrho_A = FT_density(q_A, kgrid)
    else:
        FTrho_A = np.empty(len(kgrid))
        FTrho_A[:] = np.NaN
    if e_A != e_B:
        q_B = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_B ])
        n_B = len(q_B)
        print("Number of element B: ", n_B)
        if n_B > 0:
            FTrho_B = FT_density(q_B, kgrid)
        else:
            FTrho_B = np.empty(len(kgrid))
            FTrho_B[:] = np.NaN
    else:
        FTrho_B = FTrho_A
    return np.multiply(FTrho_A, np.conjugate(FTrho_A))/n_A, \
                   (np.multiply(FTrho_A, np.conjugate(FTrho_B)))/(n_A*n_B)**0.5, \
                   np.multiply(FTrho_B, np.conjugate(FTrho_B))/n_B

def Sk_latt(names, q, kgrid, e_A, e_B, cell, bondlength=None):
    # This is the un-normalized FT for the density fluctuations
    q_A = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_A ])
    n_A = len(q_A)
    print("Number of element A: ", n_A)
    if n_A > 0:
        FTrho_A = FT_density(q_A, kgrid)
    else:
        FTrho_A = np.empty(len(kgrid))
        FTrho_A[:] = np.NaN
    if e_A != e_B:
        q_B = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_B ])
        n_B = len(q_B)
        print("Number of element B: ", n_B)
        if n_B > 0:
            FTrho_B = FT_density(q_B, kgrid)
        else:
            FTrho_B = np.empty(len(kgrid))
            FTrho_B[:] = np.NaN
        if bondlength is None:
            all_bondlength = []
            for i in range(n_A):
                for j in range(n_B):
                    bond = (q_B[j] - q_A[i])*cell
                    _bondlength = np.linalg.norm(bond)
                    if _bondlength < 2.5:
                        all_bondlength.append(_bondlength)
            all_bondlength = np.array(all_bondlength)
            bondlength = np.mean(all_bondlength)
        FTrho_cross = FT_density_latt(q_A, q_B, kgrid, [bondlength/cell[0], bondlength/cell[1], bondlength/cell[2]])
    return np.multiply(FTrho_A, np.conjugate(FTrho_A))/n_A, \
                   (FTrho_cross)/(n_A*n_B)**0.5, \
                   np.multiply(FTrho_B, np.conjugate(FTrho_B))/n_B, bondlength

def FT_density_latt(q_A, q_B, kgrid, bond):
    # This is the un-normalized FT for density fluctuations
    ng = len(kgrid)
    ak = np.zeros(ng,dtype=complex)
    ak_A = np.zeros(ng,dtype=complex)
    ak_B = np.zeros(ng,dtype=complex)
    for n,k in enumerate(kgrid):
        ak_A[n] = np.sum(np.exp(-1j*(q_A[:,0]*k[0]+q_A[:,1]*k[1]+q_A[:,2]*k[2])))
        ak_B[n] = np.sum(np.exp(-1j*((q_B[:,0]+bond[0])*k[0]+(q_B[:,1]+bond[1])*k[1]+(q_B[:,2]+bond[2])*k[2])))
    ak = np.multiply(ak_A, np.conjugate(ak_B))
    return ak

def FT_density(q, kgrid):
    # This is the un-normalized FT for density fluctuations
    ng = len(kgrid)
    ak = np.zeros(ng,dtype=complex)

    for n,k in enumerate(kgrid):
        ak[n] = np.sum(np.exp(-1j*(q[:,0]*k[0]+q[:,1]*k[1]+q[:,2]*k[2])))
    return ak

# def main(sprefix="Sk", straj="traj", sbins=14):
def main(sbins=14, _startframe=1000, _incrementframe=1, _elem1=1, _elem2=2, _dkfrac=1.0, _endframe=None):
    sprefix="Sk"
    straj="lj"

    # the input file
    print("Reading file:", straj,".lammpstrj")
    traj = open(straj+'.lammpstrj',"r")
    # number of k grids
    bins = int(sbins)
    startframe = int(_startframe)
    incrementframe = int(_incrementframe)
    elem1 = _elem1
    elem2 = _elem2
    dkfrac = float(_dkfrac)
    # get total number of bins and initialize the grid
    print("Use number of bins:", bins)
    print(f"startframe = {startframe}, increment = {incrementframe}")

    # Outputs
    ofile_AA = open(sprefix+'-'+elem1+elem1+'-real.dat',"wb")
    ofile_AB = open(sprefix+'-'+elem1+elem2+'-real.dat',"wb")
    ofile_BB = open(sprefix+'-'+elem2+elem2+'-real.dat',"wb")

    nframe = 0
    bondlength = None
    while True:
        start_time = time.time()
        # read frame
        try:
            [ cell, names, sq] = read_lammpstrj(traj)
        except:
            print("End of file, nframe = ", nframe)
            break
        print(f"Reading Frame {nframe}")

        if (nframe == 0):
            print(f"    reading kgrid")
            # normalization
            volume = np.prod(cell[:])

            kgrid = np.zeros((bins*bins*bins,3),float)
            kgridbase = np.zeros((bins*bins*bins,3),float)
            kgrid1D = np.zeros((bins*bins*bins),float)
            # initialize k grid
            cutoff_wavenumber = 1./2.4
            [ dkx, dky, dkz ] = [ dkfrac/cell[0], dkfrac/cell[1], dkfrac/cell[2] ]
            # [ dkx, dky, dkz ] = [min(cutoff_wavenumber/bins,1./cell[0]), min(cutoff_wavenumber/bins,1./cell[1]), min(cutoff_wavenumber/bins,1./cell[2]) ,]
            n=0
            for i in range(bins):
                for j in range(bins):
                    for k in range(bins):
                        if i+j+k == 0: pass
                        # initialize k grid
                        k_now = [ dkx*i, dky*j, dkz*k ]
                        # if np.linalg.norm(k_now) > cutoff_wavenumber:
                        #     print("k too large: ",i,j,k,k_now," > ", cutoff_wavenumber)
                        #     continue
                        kgridbase[n,:] = (2.*pi)*np.array([i, j, k])
                        kgrid[n,:] = [ dkx*i, dky*j, dkz*k ]
                        kgrid1D[n] = np.linalg.norm(np.array(kgrid[n]))*np.linalg.norm(np.array(kgrid[n]))
                        n+=1
            print(kgrid[-1])
            print(kgrid1D[-1])
            np.savetxt(sprefix+'-kgrid.dat',kgrid)
            np.savetxt(sprefix+"-kgrid1D.dat",kgrid1D)

        nframe += 1
        if nframe % incrementframe != 0:
            continue
        if nframe < startframe:
            continue
        print("Frame No:", nframe)
        print("--- %s seconds after read frame ---" % (time.time() - start_time))
        # FT analysis of density fluctuations
        # sk_AA, sk_AB, sk_BB, bondlength = Sk_latt(names, sq, kgridbase, ["1"], ['3'], cell, bondlength)
        sk_AA, sk_AB, sk_BB = Sk(names, sq, kgridbase, [elem1], [elem2])
        print("--- %s seconds after FFT density" % (time.time() - start_time))

        # Outputs
        np.savetxt(ofile_AA,sk_AA[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_AB,sk_AB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_BB,sk_BB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        ofile_AA.flush()
        ofile_AB.flush()
        ofile_BB.flush()

    print("A total of data points ", nframe)

    sys.exit()


if __name__ == '__main__':
    print(*sys.argv[1:])
    main(*sys.argv[1:])

# to use: python ./get-sk-3d.py [inputfile] [outputfile] [nbin]
