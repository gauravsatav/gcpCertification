# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:19:35 2022

@author: GauravSatav
"""




import bs4
import requests
import pandas as pd
import re
from time import sleep

baseUrl = "https://www.cloudskillsboost.google/course_templates/"
#finalUrl = baseUrl+str(pageNumber)
courseNumber = ['3','10','8','12','11','9','17','18','40','39','158','191','117','126','183']

courses=list()
for a in courseNumber:
    page = requests.get(baseUrl+a)
    soup = bs4.BeautifulSoup(page.content,'html')
    text = str(soup.select('ql-course'))
    text = text.split('expanded')
    sectionSylabus = []
    
    pattern1 = '&quot;title&quot;:&quot;([\w\s]+)&quot;'
    pattern2 = '"title":"([\w|\s]+)"'
    
    finalPattern=pattern1
    
    if len(re.findall(pattern1,text[0]))>1:
        finalPattern = pattern1
    elif len(re.findall(pattern2, text[0]))>1:
        finalPattern = pattern2
    else:
        print("Unkown pattern for = "+ a)
    
    for section in text[:-1]:
        sectionSylabus.append(re.findall(finalPattern,section))
    courses.append(sectionSylabus)

q = ''
for section in courses:
    q = q + '\n------------\n'
    for course in section:
            q = q+'\n\t* '
            q = q+'\n\t\t* '.join(course)

