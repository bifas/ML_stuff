from datetime import datetime, date, time
d_raw = "2018-10-07T15:14:00Z"
d_raw2 = "2019-10-07T15:14:00Z"
sum_13 = 0
sum_31 = 0
for i in range(0,1000,13):
    sum_13+=i
    print(i)
for i in range(0,1000,31):
    sum_31+=i
print(sum_13+sum_31)

# date1 = datetime.strptime(d_raw, "%Y-%m-%dT%H:%M:%SZ")
# #date2 =datetime.strptime(d_raw2, "%Y-%m-%dT%H:%M:%SZ")
# date2 = "a"
# if not date2: #the bigger is the departure
#     print(1)
# else:
#     print(2)
#
# x = sum(n/2 for n in range(2,6,2))
# print(x)
# x = 10
# y = 15
# x = x+y
# y = x-y
# x = x-y
# print(x,y)
#
# i = 0
# while i<3:
#     print(++i)
#     i +=1
#     print(i+1)
#
# print(list("hello"))
#
# l = [1,2,3,4,5,1,2,3,4,5,0,1]
# print(len(set(l))/2)
#
# l = (1,2,5,6,4,3)
# print(l[int(-1/2):int(7/2)])