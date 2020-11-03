# -*- coding: utf-8 -*-
import os
import csv

pathM =r"C:\Users\peter fazekas\Desktop\nofakes project\FALdetector-master\val\modified"
path =r"C:\Users\peter fazekas\Desktop\nofakes project\FALdetector-master\val\original"
with open('train.csv', 'w', newline='') as csvfile: #Loop through path and add all files matching *.jpg to array files
    files = []
    for r,d,f in os.walk(path):
        for _file in f:
            if '.png' in _file:
                files.append(_file)
    
    writer = csv.writer(csvfile, delimiter=',') #Create a writer from csv module
    for f in files: #find type of file
        t=0 #cut off the number and .jpg from file, leaving only the type (this may have to be changed.)
        writer.writerow([f, t]) #write the row to the file output.csv
        
    files = []
    for r,d,f in os.walk(pathM):
        for _file in f:
            if '.png' in _file:
                files.append(_file)
    
    writer = csv.writer(csvfile, delimiter=',') #Create a writer from csv module
    for f in files: #find type of file
        t=1 #cut off the number and .jpg from file, leaving only the type (this may have to be changed.)
        writer.writerow([f, t]) #write the row to the file output.csv
