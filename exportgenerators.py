k=1;s=range(1,n+1);s[k-1],s[k]=s[k],s[k-1];np.savetxt('/Users/solver/Desktop/'+str(k)+'.csv',spc(s),delimiter=",")



l=[3,2,1]
spc=SymmetricGroupRepresentation(l,"specht");n=sum(l);
for k in range(1,n):
    s=range(1,n+1);s[k-1],s[k]=s[k],s[k-1]
    gen=spc(s)
    np.savetxt('/tmp/'+str(k)+'.csv',gen,delimiter=",")
    
