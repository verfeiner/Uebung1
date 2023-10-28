import random
import time
import matplotlib.pyplot as plt

def quicksort(arr,left,right):
    if left<right:
        quicksort_pos=partition(arr,left,right)
        quicksort(arr,left,quicksort_pos-1)
        quicksort(arr,quicksort_pos+1,right)

def partition(arr,left,right):
    i=left
    j=right-1
    pivot=arr[right]
    while i<j:

         while i<right and arr[i]<pivot:
            i=i+1
         while j>left and arr[j]>pivot:
            j=j-1
         if i<j:
            arr[i],arr[j]=arr[j],arr[i]
    if arr[i]>pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i

#arr=[12,20,2,45,111,9,1,22]
#quicksort(arr,0,len(arr)-1)
#print(arr)
#c=len(arr)
#print(c)
#print(arr[7])

def iqr(arr):
    n=len(arr)
    if n%4==0:
       n1=int(n/4)-1
       n3=int(n*0.75)-1
       q1=0.5*(arr[n1]+arr[n1+1])
       q3=0.5*(arr[n3]+arr[n3+1])# 数组后的数字必须设定成int整数，数组是从0开始计算的
       #print(n1,n3)
    else:
       n4=int(n/4)
       n5=int(n * 0.75)
       q1=arr[n4+1]
       q3 = arr[n5 + 1]
    #print(q1,q3)

    return q3-q1

#IQR=iqr(arr)

#print(IQR)

runtimes=[]


arr_size=[10,20,30,100]
for t in arr_size:
    arr=[random.randint(1,1000) for _ in range(t)]
    start_time =time.time() #
    quicksort(arr, 0,len(arr) - 1)#quicksort只是排序，排序完数组没有任何变化，变量名也是不变
    print(arr)
    IQR = iqr(arr)
    #print(IQR)
    end_time = time.time()
    #print(end_time-start_time)
    runtimes.append(end_time-start_time)
    #print(runtimes)#命名重复容易出错
    print(f"Data Size: {t}, IQR: {IQR}")

plt.plot(arr_size,runtimes)
plt.ylim(0,0.1)
plt.xlabel("data_size")
plt.ylabel("runtime")
plt.show()