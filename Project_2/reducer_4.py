#!/usr/bin/python3

import sys

emp_dict ={}
salary_dict={}
for line in sys.stdin:
    line = line.strip()
    EmployeeID, Name, Salary, Country, Passcode = line.split('\t')
    

    if Passcode =='-1':
        emp_dict[EmployeeID] = Name
    else:
        salary_dict[EmployeeID] = [Salary,Country,Passcode]

#print(emp_dict,salary_dict)
for EmployeeID in salary_dict.keys():
    Name = emp_dict[EmployeeID]
    Salary = salary_dict[EmployeeID][0]
    Country = salary_dict[EmployeeID][1] 
    Passcode = salary_dict[EmployeeID][2] 

    print ('%s\t%s\t%s\t%s\t%s'% (EmployeeID,Name,Salary,Country,Passcode))
