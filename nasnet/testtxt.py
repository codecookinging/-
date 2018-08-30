
a=[1.11,2.22222,3.333]
b=['a','acc','ab']
c=['uu','a','b']
with open('2.txt','w') as f:
    for i in range(len(a)):


        f.write('{:<50}'.format(str(a[i])))
        #f.write(str(a[i]))
        f.write('{:<80}'.format((b[i])))
        #f.write(b[i])

        f.write(c[i])
        f.write('\n')



