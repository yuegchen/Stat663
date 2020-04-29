# Stat663
Final project of Stat663.This project is an implementation of partial least squre regression algorithm as described in Partial Least-Squares Regression: A Tutorial by Paul Geladi and Bruce R. Kowalsiki in 1985.

This package can be installed by 
"python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps Stat_663_Plsr"
or
"pip3 install -i https://test.pypi.org/simple/ Stat-663-Plsr==0.0.1"

This implementation contains three different versions of plsr algorithm: 1. plain python 2. jit optimized version 3. cythonlized version. The execution and comparison of running speed of all 3 versions can be found in example/fp.ipynb

It can be imported by "import plsr" and to use cythonlized functions "from plsr.plsr_cy.plsr_cython import plsr_cython,predict_cython". Detailed use cases can be found in test directory. Note: cythonlied module in the package is of different usage than the cythonlied method used in ipynb file.


Execution of the algorithm on the real-world data can be found in example/plsr_test_real_data.ipynb.