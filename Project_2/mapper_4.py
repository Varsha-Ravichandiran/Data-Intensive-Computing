#!/usr/bin/python3
 
import sys
 
for line in sys.stdin:
         
        EmployeeID = "-1"
        Name = "-1" 
        Salary = "-1"
        Country = "-1" 
        Passcode = "-1" 
        
        line = line.strip()
         
        splits = line.split(",")
        if len(splits)== 4:
            continue
        if len(splits) == 2:
            EmployeeID = splits[0]
            Name = splits[1]
        elif len(splits) == 5:
             EmployeeID = splits[0]
             Salary = splits[1]+','+splits[2]
             Country = splits[3]
             Passcode = splits[4]  
        else:
            EmployeeID = splits[0]
            Salary = splits[1]+','+splits[2]
            Country = splits[3]+','+splits[4]
            Passcode = splits[5]       
         
        print ('%s\t%s\t%s\t%s\t%s' % (EmployeeID,Name,Salary,Country,Passcode))
