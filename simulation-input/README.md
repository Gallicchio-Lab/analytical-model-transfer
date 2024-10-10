# Simulation inputs to run AToM-OpenMM

Example for temoa-g1

## Coupling, hydration and transfer:
Make symbolic link of sys.xml file of the system in the complex directory. 
```
ln -s temoa-g1-z_sys.xml hydration/temoa-g1-z_sys.xml
```

## Coupling and transfer:
Make symbolic link of the PDB file of the system. 
```
ln -s temoa-g1-z.pdb coupling/temoa-g1-z.pdb
```

## To run AToM-OpenMM:
```
python /AToM-OpenMM/abfe_explicit.py temoa-g1_transfer_asyncre.cntl
```
