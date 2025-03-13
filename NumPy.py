#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

x=np.array([1,2,3,4])  
print(x)  # this is array


# In[2]:


y=[1,2,3,4]
print(y)  # this is list


# In[3]:


import numpy as np

x=np.array([1,2,3,4])  
print(x)  # this is array
print(type(x))


# In[4]:


y=[1,2,3,4]
print(y)  # this is list
print(type(y))


# # NumPy arrays vs Lists

# In[9]:


get_ipython().run_line_magic('timeit', '[j**4 for j in range(1,9)]   # Slower')


# In[7]:


import numpy as np

get_ipython().run_line_magic('timeit', 'np.arange(1,9)**4   # Faster')


# # Creating NumPy Arrays

# In[10]:


import numpy as np

x=[1,2,3,4]  

y=np.array(x)  # Method 1

print(y)  


# In[11]:


import numpy as np

y=np.array([1,2,3,4]) # Method 2

print(y)


# In[12]:


import numpy as np

y=np.array([1,2,3,4])   

print(y)

y


# In[13]:


import numpy as np

y=np.array([1,2,3,4])   

print(y)

print(type(y))


# In[14]:


l=[]                          # take input from user

for i in range(1,5):
    int_1=input("enter : ")   # the problem here is the array is in string
    l.append(int_1)
    
print(np.array(l))    


# In[15]:


l=[]

for i in range(1,5):
    int_1=int(input("enter : "))   # array is in integer
    l.append(int_1)
    
print(np.array(l)) 


# # Dimensions in Arrays

# In[1]:


import numpy as np

y=np.array([1,2,3,4])   

print(y)

print(type(y))
print(y.ndim)


# In[4]:


ar2=np.array([[1,2,3,4],[1,2,3,4]])
print(ar2)
print(ar2.ndim)


# In[8]:


ar3=np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
print(ar3)
print(ar3.ndim)


# In[12]:


arn = np.array( [1,2,3,4] , ndmin=10 )  # for n no.of dimensions. Here n=10
print(arn)
print(arn.ndim)


# # Create NumPy Array Using NumPy functions

# # (i) Zeros

# In[13]:


import numpy as np

ar_zero=np.zeros(4)

print(ar_zero)


# In[15]:


import numpy as np

ar_zero1=np.zeros((3,4))

print(ar_zero1)


# # (ii) Ones

# In[16]:


ar_one=np.ones(4)
print(ar_one)


# # (iii) Empty

# In[17]:


ar_em=np.empty(4)
print(ar_em)       # Previous memory's data come into empty array


# # (iv) Range 

# In[18]:


ar_rn=np.arange(4)
print(ar_rn)


# # (v) Diagonal

# In[19]:


ar_dia=np.eye(3)
print(ar_dia)


# In[20]:


ar_dia=np.eye(3,5)
print(ar_dia)


# # (vi) linspace 

# In[24]:


ar_lin=np.linspace(1,10,num=5)   # range is 1 to 10 and no.of elements is 5
print(ar_lin)               


# In[22]:


ar_lin=np.linspace(0,10,num=5)   
print(ar_lin)               


# In[23]:


ar_lin=np.linspace(0,20,num=5)   
print(ar_lin) 


# # Create NumPy Arrays with Random Numbers                                                                                                                            

# # (i) rand()

# In[26]:


import numpy as np

var = np.random.rand(4)

print(var)       # all values range from 0 to 1


# In[27]:


import numpy as np

var1 = np.random.rand(2,5)

print(var1)  


# # (ii) randn()

# In[28]:


import numpy as np

var2 = np.random.randn(5)

print(var2)              # values close to zero , may be +ve or -ve as well


# # (iii) ranf()

# In[2]:


import numpy as np

var3 = np.random.ranf(4)

print(var3)        # range is [0.0,1.0)


# # (iv) randint()

# In[32]:


import numpy as np

var4 = np.random.randint(5,20,5)    #  np.random.randint(min,max,total_no._of_values_generated)

print(var4) 


# # Data Types

# In[34]:


import numpy as np

var=np.array([1,2,3,4,12,13,14,15])

print("Data Type : " ,var.dtype)


# In[36]:


var=np.array([1.0,1.2,3,4,12,13,14,15])

print("Data Type : " ,var.dtype)


# In[37]:


var=np.array(["a","b","c","d"])

print("Data Type : " ,var.dtype)   # string data type


# In[3]:


var=np.array(["a","b","c","d",1,2,3,4])

print("Data Type : " ,var.dtype)  # string and integer data type


# In[4]:


x=np.array([1,2,3,4])
print("Data Type : " ,x.dtype)   # we will convert it's data type


# In[5]:


x=np.array([1,2,3,4],dtype=np.int8)   # we have converted it's data type to int8
print("Data Type : " ,x.dtype)       


# In[6]:


x=np.array([1,2,3,4],dtype=np.int8)   
print("Data Type : " ,x.dtype)
print(x)


# In[7]:


x1=np.array([1,2,3,4],dtype="f")   # convert into float32
print("Data Type : " ,x1.dtype)
print(x1)


# In[9]:


x2=np.array([1,2,3,4]) 

new=np.float32(x2)       # Convert it from int to float.

print("Data Type : " ,x2.dtype)
print("Data Type : " ,new.dtype)

print(x2)
print(new)


# In[10]:


x2=np.array([1,2,3,4]) 

new=np.float32(x2)

new_one=np.int_(new)   # Convert it from float to int again.

print("Data Type : " ,x2.dtype)
print("Data Type : " ,new.dtype)
print("Data Type : " ,new_one.dtype)   

print(x2)
print(new)
print(new_one)


# In[11]:


x3=np.array([1,2,3,4]) 

new_1=x3.astype(float)   # direct conversion from int to float

print(x3)
print(new_1)


# # Arithmetic Operations

# # (i) In 1D array

# In[12]:


import numpy as np

var=np.array([1,2,3,4])    # using normal method

varadd= var+3

print(varadd)


# In[21]:


import numpy as np

var=np.array([1,2,3,4])

varadd= np.add(var,3)    # using numpy operator

print(varadd)


# In[13]:


var1=np.array([1,2,3,4])
var2=np.array([1,2,3,4])

varadd= var1+var2    # using normal method

print(varadd)


# In[20]:


var1=np.array([1,2,3,4])
var2=np.array([1,2,3,4])

varadd= np.add(var1,var2)    # using numpy operator

print(varadd)


# In[14]:


import numpy as np

var=np.array([1,2,3,4])

varsubtract= var-3

print(varsubtract)


# In[16]:


import numpy as np

var=np.array([1,2,3,4])

varmultiply= var*3

print(varmultiply)


# In[18]:


import numpy as np

var=np.array([1,2,3,4])

vardivide= var/3

print(vardivide)


# In[19]:


import numpy as np

var=np.array([1,2,3,4])

varmod= var%3
                       
print(varmod)


# In[27]:


import numpy as np

var=np.array([1,2,3,4])

varreciprocal=np.reciprocal(var)
                       
print(varreciprocal)


# # (ii) In 2D array

# In[28]:


import numpy as np

var21=np.array([[1,2,3,4],[1,2,3,4]])
var22=np.array([[1,2,3,4],[1,2,3,4]])

print(var21)
print()         # for gap
print(var22)
print()         # for gap   

varadd2=np.add(var21,var22) #var21+var22  
                       
print(varadd2)


# In[26]:


import numpy as np

var21=np.array([[1,2,3,4],[1,2,3,4]])
var22=np.array([[1,2,3,4],[1,2,3,4]])

print(var21)
print()         # for gap
print(var22)
print()         # for gap   

varmultiply2= var21*var22
                       
print(varmultiply2)


# # Arithmetic Functions

# In[29]:


import numpy as np

var=np.array([1,2,3,4,1,15,0,6])

print("min : ",np.min(var))
print("max : ",np.max(var))


# In[36]:


import numpy as np

var=np.array([1,2,3,4,1,15,0,6])

print("min : ",np.min(var),np.argmin(var))  # min and max values with index positions
print("max : ",np.max(var),np.argmax(var))
print("sqrt : ",np.sqrt(var))


# In[32]:


var1=np.array([[2,7,3],[6,5,9]])

print(np.min(var1,axis=0))


# In[34]:


var1=np.array([[2,7,3],[6,5,9]])

print(np.min(var1,axis=1))


# In[33]:


var1=np.array([[2,7,3],[6,5,9]])

print(np.max(var1,axis=0))


# In[35]:


var1=np.array([[2,7,3],[6,5,9]])

print(np.max(var1,axis=1))


# In[39]:


var2=np.array([1,2,3])

print(np.sin(var2))
print(np.cos(var2))
print(np.cumsum(var2))


# # Shape in Arrays

# In[2]:


import numpy as np

var=np.array([[1,2,3],[1,2,3]])

print(var)
print()
print(var.shape)


# In[3]:


import numpy as np

var=np.array([[1,2,3,4],[1,2,3,4]])

print(var)
print()
print(var.shape)


# In[8]:


var1=np.array([1,2,3,4],ndmin=4)

print(var1)
print()
print(var1.shape)    # no.of rows is 3 i.e., 1,1,1 & no.of cols is 4  ---> (1,1,1,4)
print()
print(var1.ndim)


# # Reshape in Arrays

# In[12]:


var2=np.array([1,2,3,4,5,6])

print(var2)
print(var2.ndim)
print()
                      # 1D array to 2D array.
x=var2.reshape(3,2)   # rows-->3 & columns-->2

print(x)
print(x.ndim)


# In[13]:


var2=np.array([1,2,3,4,5,6])

print(var2)
print(var2.ndim)
print()

x=var2.reshape(3,3)   # rows-->3 & columns-->3 is not possible as we don't have enough elements.

print(x)
print(x.ndim)


# In[16]:


var2=np.array([1,2,3,4,5,6,7,8,9])

print(var2)
print(var2.ndim)
print()
                      # 1D array to 2D array.
x=var2.reshape(3,3)   # rows-->3 & columns-->3

print(x)
print(x.ndim)


# In[17]:


var3=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

print(var3)
print(var3.ndim)
print()

x=var3.reshape(2,3,2)   # 1D array to 3D array.
                        # 2-->elements, 3-->rows in each element & 2-->columns in each element.
print(x)
print(x.ndim)


# In[19]:


var3=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

print(var3)
print(var3.ndim)
print()

x=var3.reshape(2,3,2)   # 1D array to 3D array.
                        # 2-->elements, 3-->rows in each element & 2-->columns in each element.
print(x)
print(x.ndim)
print()

one_D=x.reshape(-1)     # 3D to 1D array.
print(one_D)         
print(one_D.ndim)


# # Broadcasting 

# In[20]:


import numpy as np

var1=np.array([1,2,3,4])

var2=np.array([1,2,3])

print(var1+var2)      # error is showing because no.of dimensions is not equal


# In[21]:


import numpy as np

var1=np.array([1,2,3,4])

var2=np.array([1,2,3,4])

print(var1+var2)      # no error is showing because no.of dimensions is equal


# In[31]:


import numpy as np

var1=np.array([1,2,3])
print(var1)
print()
print(var1.shape)
print()

var2=np.array([[1],[2],[3]])
print(var2)
print()
print(var2.shape)
print()

print(var1+var2)     # broadcasting 
      


# In[38]:


x=np.array([[1],[2]])
print(x)
print(x.shape)
print()

y=np.array([[1,2,3],[1,2,3]])
print(y)
print(y.shape)
print()

print(x+y)


# # Indexing

# In[2]:


import numpy as np

var=np.array([9 , 8 , 7 , 6])  # 1D array
#             0 , 1 , 2 , 3
#            -4 ,-3 , -2 ,-1 

print(var[1])
print(var[-3])       


# In[46]:


var1=np.array([[9,8,7],[4,5,6]])   # 2D array

print(var1)
print(var1.ndim)
print()

print(var1[0,2])


# In[51]:


var2=np.array([[[1,2],[6,7]]])
print(var2)
print(var2.ndim)
print()
print(var2[0,1,1])


# # Slicing

# In[52]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:4])


# In[54]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:5])


# In[55]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:4])    # last index is excluded 

print("2 to End : ",var[1:])


# In[57]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:4])

print("2 to End : ",var[1:])

print("start to 5 : ",var[:5])


# In[58]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:4])

print("2 to End : ",var[1:])

print("start to 5 : ",var[:5])

print("stop: ",var[::2])


# In[60]:


var=np.array([1,2,3,4,5,6,7])
#             0 1 2 3 4 5 6

print(var)

print()

print("2 to 5 : ",var[1:4])   

print("2 to End : ",var[1:])

print("start to 5 : ",var[:5])

print("stop: ",var[::2])  # step is 2 from start to end

print("stop: ",var[1:5:2])  # step is 2 from index1(2) to index4(5)


# In[64]:


var1=np.array([[1,2,3,4,5],[9,8,7,6,5],[11,12,13,14,15]])    # for 2D array
print(var1)
print()
print("8 to 5 : ",var1[1,1:])


# In[66]:


var1=np.array([[1,2,3,4,5],[9,8,7,6,5],[11,12,13,14,15]])    # for 2D array
print(var1)
print()
print("12 to 15 : ",var1[2,1:])


# # Iteration

# In[67]:


var=np.array([9,8,7,6,5,4])
print(var)
print()

for i in var:
    print(i)


# In[70]:


var1=np.array([[1,2,3,4],[1,2,3,4]])          
print(var1)                              
print()

for j in var1:
    print(j)
    
print()

for k in var1:       # for iterating 
    for l in k:
        print(l)      


# In[6]:


var3=np.array([[[1,2,3,4],[1,2,3,4]]])   # 3D array
print(var3)
print(var3.ndim)
print()

for i in var3:          # using for loop
    for j in i:
        for k in j:
            print(k)


# In[7]:


var3=np.array([[[1,2,3,4],[1,2,3,4]]])   # 3D array
print(var3)
print(var3.ndim)
print()

for i in np.nditer(var3):    # using nditer
    print(i)


# In[8]:


var3=np.array([[[1,2,3,4],[1,2,3,4]]])   # 3D array
print(var3)
print(var3.ndim)
print()

for i in np.nditer(var3,flags=['buffered'],op_dtypes=["S"]):    # using nditer and changing dtypes fron int to string
    print(i)


# In[12]:


var3=np.array([[[1,2,3,4],[1,2,3,4]]])   
print(var3)
print(var3.ndim)
print()

for i,d in np.ndenumerate(var3):    # iterated values with index numbers for 3D array
    print(i,d)


# # Copy vs View

# In[13]:


var=np.array([1,2,3,4])

co=var.copy()

print("var :",var)
print("copy : ",co)


# In[14]:


x=np.array([9,8,7,6])

vi=x.view()

print("x : ",x)
print("view : ",vi) 


# In[15]:


var=np.array([1,2,3,4])

co=var.copy()

var[1]=40

print("var :",var)
print("copy : ",co)   # changes made in var(original data) don't affect copy data


# In[16]:


x=np.array([9,8,7,6])

vi=x.view()

x[1]=40

print("x : ",x)
print("view : ",vi) # changes made in x(original data) affect the view data


# # Join Array

# In[17]:


var=np.array([1,2,3,4])
var1=np.array([9,8,7,6])

ar=np.concatenate((var,var1))   # joining two 1D arrays

print(ar)


# In[18]:


vr=np.array([[1,2],[3,4]])
vr1=np.array([[9,8],[7,6]])

ar_new=np.concatenate((vr,vr1),axis=1)   # joining two 2D arrays along axis1

print(vr)
print()
print(vr1)
print()
print(ar_new)


# In[19]:


vr=np.array([[1,2],[3,4]])
vr1=np.array([[9,8],[7,6]])

ar_new=np.concatenate((vr,vr1),axis=0)   # joining two 2D arrays along axis0

print(vr)
print()
print(vr1)
print()
print(ar_new)


# In[25]:


var_1=np.array([1,2,3,4])
var_2=np.array([9,8,7,6])

a_new=np.stack((var_1,var_2),axis=0)   # joining two 1D arrays using stack function
                                       # axis 0 --> horizontal wise  axis 1--> vertical wise
print(a_new)


# In[26]:


var_1=np.array([1,2,3,4])
var_2=np.array([9,8,7,6])

a_new=np.stack((var_1,var_2),axis=1)   # joining two 1D arrays using stack function
                                       # axis 0 --> horizontal wise  axis 1--> vertical wise
print(a_new)


# In[27]:


var_1=np.array([1,2,3,4])
var_2=np.array([9,8,7,6])

a_new=np.hstack((var_1,var_2))   # joining two 1D arrays using stack function
                                 # np.hstack--> horizontal wise  / row
print(a_new)


# In[28]:


var_1=np.array([1,2,3,4])
var_2=np.array([9,8,7,6])

a_new=np.vstack((var_1,var_2))   # joining two 1D arrays using stack function
                                 # np.vstack --> vertical wise / column 
print(a_new)


# In[29]:


var_1=np.array([1,2,3,4])
var_2=np.array([9,8,7,6])

a_new=np.dstack((var_1,var_2))   # joining two 1D arrays using stack function
                                 # np.dstack --> depth wise / height
print(a_new)


# # Split Array

# In[2]:


import numpy as np

var=np.array([1,2,3,4,5,6])    # 1D array

print(var)

ar=np.array_split(var,3)

print()

print(ar)
print(type(ar))


# In[3]:


var=np.array([1,2,3,4,5,6])  # 1D array

print(var)

ar=np.array_split(var,3)

print()

print(ar)
print(type(ar))
print(ar[0])     # to get only one element of the list


# In[5]:


var1=np.array([[1,2],[3,4],[5,6]])   # 2D array

print(var1)

ar=np.array_split(var1,3)

print()

print(ar)
print(type(ar))


# In[6]:


var1=np.array([[1,2],[3,4],[5,6]])   # 2D array

print(var1)

ar1=np.array_split(var1,3)
ar2=np.array_split(var1,3,axis=1)

print()

print(ar1)
print(type(ar1))
print()

print(ar2)
print(type(ar2))


# # Search Array

# In[7]:


var=np.array([1,2,3,4,5,6,7])
#index        0 1 2 3 4 5 6

x=np.where( var==2)
print(x)


# In[9]:


var=np.array([1,2,3,4,5,6,7])
#index        0 1 2 3 4 5 6

x=np.where((var%2)==0)
print(x)


# # Search Sorted Array

# In[10]:


var1=np.array([1,2,3,4,6,7,8])   # this must be a sorted array
#index         0 1 2 3 4 5 6 

x1=np.searchsorted(var1,5)  # it places the element in it's index position according to the array (left)
print(x1)


# In[12]:


var1=np.array([1,2,3,4,6,7,8,9,10])   # this must be a sorted array
#index         0 1 2 3 4 5 6 7  8 

x1=np.searchsorted(var1,5,side="right")  # it places the element from right
print(x1)


# In[13]:


var1=np.array([1,2,3,4,6,7,8,9,10])   # this must be a sorted array
#index         0 1 2 3 4 5 6 7  8 

x1=np.searchsorted(var1,[5,6,7],side="right")  # it places the element from right
print(x1)


# # Sort Array

# In[14]:


var_1=np.array([4,3,2,1,12,5,22,52,6,9])  # sorting 1D array

print(np.sort(var_1))


# In[15]:


var_2=np.array(["a","s","d","t","c","m"])

print(np.sort(var_2))


# In[16]:


var_1=np.array([[4,3,2],[1,12,5],[22,52,6]])   # sorting 2D array

print(np.sort(var_1))


# # Filter Array

# In[18]:


var_3=np.array(["a","s","d","t","c","m"])

f=[True,False,True,True,False,False]

new_a=var_3[f]

print(new_a)
print(type(new_a))


# # Shuffle

# In[21]:


var=np.array([1,2,3,4,5])

np.random.shuffle(var)

print(var)


# # Unique

# In[23]:


var_1=np.array([1,2,3,4,5,8,1,2,5,9,10])

x=np.unique(var_1)    # gives unique values 


print(x)


# In[24]:


var_1=np.array([1,2,3,4,5,8,1,2,5,9,10])

x=np.unique(var_1,return_index=True)  # gives unique values with index numbers

print(x)


# In[25]:


var_1=np.array([1,2,3,4,5,8,1,2,5,9,10])

x=np.unique(var_1,return_index=True,return_counts=True)  # gives unique values with index numbers with their counts

print(x)


# # Resize

# In[29]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(2,3))   # rows-->2 and columns-->3

print(y)             


# In[28]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   # rows-->3 and columns-->2

print(y) 


# # Flatten 

# In[32]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(2,3))   

print(y)
print()
print(y.flatten())             


# In[35]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print(y.flatten(order="F")) 


# In[36]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print(y.flatten(order="C"))


# # Ravel

# In[38]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print("Flatten : ",y.flatten(order="F"))
print("Ravel : ",np.ravel(y))


# In[39]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print("Flatten : ",y.flatten(order="F"))
print("Ravel : ",np.ravel(y,order="F"))


# In[40]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print("Flatten : ",y.flatten(order="F"))
print("Ravel : ",np.ravel(y,order="A"))


# In[41]:


var2=np.array([1,2,3,4,5,6])

y=np.resize(var2,(3,2))   

print(y)
print()
print("Flatten : ",y.flatten(order="F"))
print("Ravel : ",np.ravel(y,order="K"))


# # Insert Function

# In[44]:


var=np.array([1,2,3,4])

print(var)
print(var.dtype)
print(type(var))

v=np.insert(var,2,40)   # 2=position and 40=value to be inserted
print(v)


# In[45]:


var=np.array([1,2,3,4])

print(var)
print(var.dtype)
print(type(var))

v=np.insert(var,(2,4),40)    # 2,4=positions and 40=value to be inserted
print(v)


# In[46]:


var=np.array([1,2,3,4])

print(var)
print(var.dtype)
print(type(var))

v=np.insert(var,(2,4),4.5)    # it doesn't accept float value
print(v)


# In[47]:


var=np.array([1,2,3,4])

print(var)
print(var.dtype)
print(type(var))

v=np.insert(var,(2,4),6.5)    # it doesn't accept float value
print(v)


# In[54]:


var1=np.array([[1,2,3],[1,2,3]])  # 2D array
print(var1)
print()

v1=np.insert(var1,2,6,axis=0)  # inserting 1 value in 2D array 
print(v1)


# In[55]:


var1=np.array([[1,2,3],[1,2,3]])
print(var1)
print()

v1=np.insert(var1,2,6,axis=1)   # inserting 1 value in 2D array
print(v1)


# In[56]:


var1=np.array([[1,2,3],[1,2,3]])
print(var1)
print()

v1=np.insert(var1,2,[66,21],axis=1)   # inserting 2 values in 2D array
print(v1)


# In[57]:


var=np.array([1,2,3,4])    # append using 1D array

print(var)
print(var.dtype)
print(type(var))

v=np.append(var,6.5)    # we can also use append 
print(v)


# In[60]:


var1=np.array([[1,2,3],[1,2,3]])   # append using 2D array
print(var1)
print()

v1=np.append(var1,[[5,12,87]],axis=0)
print(v1)


# # Delete Function

# In[61]:


var1=np.array([1,2,3,4])

print(var1)

d=np.delete(var1,2)  # deletes data of index number 2

print(d)


# # Matrix in NumPy Arrays

# In[4]:


import numpy as np

var=np.matrix([[1,2,3],[1,2,3]])  # Matrix

print(var)
print(type(var))


# In[5]:


var1=np.array([[1,2,3],[1,2,3]])  # Array

print(var1)
print(type(var1))


# # (i) Arithmetic operations in Matrix

# In[11]:


import numpy as np

var=np.matrix([[1,2,3],[1,2,3]])  # Matrix
var2=np.matrix([[1,2,3],[1,2,3]])

print(var)
print(type(var))
print()
print(var2)
print()
print(var+var2)
print()
print(var-var2)
print()
print(var*var2) # error is showing as shape is not same
print()


# In[12]:


import numpy as np
                              # Matrix
var=np.matrix([[1,2],[1,2]])  # shape is same
var2=np.matrix([[1,2],[1,2]])

print(var)
print(type(var))
print()
print(var2)
print()
print(var+var2)
print()
print(var-var2)
print()
print(var*var2) # No error
print()


# In[13]:


import numpy as np
                              # Matrix
var=np.matrix([[1,2],[1,2]])  # shape is same
var2=np.matrix([[1,2],[1,2]])

print(var)
print(type(var))
print()
print(var2)
print()
print(var+var2)
print()
print(var-var2)
print()
print(var*var2) # No error
print()
print(var.dot(var2))   # Dot function for multiplication


# # (ii) Transpose

# In[18]:


v=np.matrix([[1,2,3],[4,5,6]])     
print(v)
print()
print(np.transpose(v)) # Transpose
print()
print(v.T)   # Short cut for Transpose


# # (iii) Swapaxes

# In[23]:


v=np.matrix([[1,2,3],[4,5,6]])     
print(v)
print()                    # swapaxes
print(np.swapaxes(v,0,1))  # converting axis 0 to axis 1


# In[24]:


v2=np.matrix([[1,2],[3,4]])
print(v2)
print()
print(np.swapaxes(v2,0,1))


# # (iv) Inverse

# In[26]:


v3=np.matrix([[1,2],[3,4]])
print(v3)
print()
print(np.linalg.inv(v3))


# # (v)Power

# In[28]:


v4=np.matrix([[1,2],[3,4]])
print(v4)
print()
print(np.linalg.matrix_power(v4,2))
print()
print(np.linalg.matrix_power(v4,0))
print()
print(np.linalg.matrix_power(v4,-2))


# # (vi) Determinant

# In[29]:


v5=np.matrix([[1,2],[3,4]])
print(v5)
print()
print(np.linalg.det(v5))
print()


# In[30]:


v6=np.matrix([[1,2,3],[3,4,3],[1,2,3]])
print(v6)
print()
print(np.linalg.det(v6))
print()


# In[ ]:




