#!/Applications/SageMath-8.1.app/sage


import numpy as np
from sage.all import *

n=len(sys.argv)-1
l=range(0,n)
for k in range(0,n):
    l[k]=int(sys.argv[k+1])

spc=SymmetricGroupRepresentation(l,"specht");n=sum(l);
for k in range(1,n):
    s=range(1,n+1);s[k-1],s[k]=s[k],s[k-1]
    gen=spc(s)
    np.savetxt('/tmp/gen'+str(k)+'.csv',gen,delimiter=",")
