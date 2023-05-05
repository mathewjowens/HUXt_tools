# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:39:28 2023

@author: vy902033
"""
import requests
import os
import json
import numpy as np
from datetime import datetime


startdate = datetime(2023, 4, 4, 0)
enddate = datetime(2023, 4, 7, 0)

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))
#where to save the output images
datadir = os.path.join(cwd, 'data')

api_key = os.getenv("API_KEY")

url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
version = "v1"

request_url = url_base+'/'+version+'/output'

#data_file = open('../output/testdata.zip', 'rb')

#response = requests.put(request_url, \
#               params = {"object_name": "testdata.zip"},\
#               headers={"Content-Type" : "application/zip", \
#                        "apikey" : api_key},\
#                        data=data_file\
#                        )

#print(response)
#print(response.content)

response = requests.get(request_url+'/list_directories', \
                 headers={"accept" : "*/*", \
                          "apikey" : api_key}
                          )
print(response)
print(response.content)





response = requests.get(request_url+'/list?directory=pfss_latest', \
                 headers={"accept" : "*/*", \
                          "apikey" : api_key}
                          )
print(response)
print(response.content)

if response.ok:
    response_dict = json.loads(response.content)
    file_list = response_dict["objects"]
else:
    print("Error:", response.status_code, response.content)


#extract the date info for each file
date_list = []
for count, filename in enumerate(file_list):
    date_time_str = filename.split("pfss")[2][0:11]
    date_list.append(datetime.strptime(date_time_str, '%Y%m%d.%H'))
    

#find the most recent date within the required date range
filtered_dates = [date for date in date_list if startdate <= date <= enddate]

if filtered_dates:
    most_recent_date = max(filtered_dates)
    print("Most recent date:", most_recent_date)
else:
    print("No dates found.")
#get the list index
index = date_list.index(most_recent_date)



#get the associated file from the API
pfss_url = request_url + "?object_name=" + file_list[index]
response_pfss = requests.get(pfss_url, headers={"accept" : "*/*", "apikey" : api_key })
pfss_filename = file_list[index].split("/")[1]
if response_pfss.status_code == 200:
                 pfssfilepath = os.path.join(datadir, pfss_filename)
                 
                 url = response_pfss.content.strip(b'"').decode('utf-8')
                 response = requests.get(url)

                 # Save the file
                 with open(pfssfilepath, 'wb') as f:
                     f.write(response.content)
                 found_pfss = True



    
# response3 = requests.get(request_url+"?object_name=pfss_latest/windbound_b_pfss20230403.11.nc", \
#                 headers={"accept" : "*/*", \
#                          "apikey" : api_key}
#                          )
# print(response3.json())

import urllib.request

headers = {
    "apikey": api_key
}

req = urllib.request.Request(pfss_url, headers=headers)
with urllib.request.urlopen(req) as response, open(pfss_filename, 'wb') as out_file:
    data = response.read() # read the response data in memory
    out_file.write(data) # write the data to file

    
# <codecell>



response3 = requests.get(request_url+"?object_name=pfss_latest/windbound_b_pfss20230403.11.nc", \
                headers={"accept" : "*/*", \
                         "apikey" : api_key}
                         )
print(response3.json())

# if response3.status_code == 200:
#     r = requests.get(response3.json(), \
#          headers={"accept" : "*/*", \
#                   "apikey" : api_key}
#     )    
#     open('testdata0.zip', 'wb').write(r.content)
#     print("Dumfric data downloaded.")
# else:
#     print("An error occured: No vaild data response.")
    
# <codecell>
import datetime
startdate = datetime.datetime(2022,1,1,9,0,0)
enddate= datetime.datetime(2023,12,30,9,0,0)

    
version = 'v1'
api_key = os.getenv("API_KEY")
url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"

startdatestr = startdate.strftime("%Y-%m-%dT%H:%M:%S")
enddatestr = enddate.strftime("%Y-%m-%dT%H:%M:%S")

request_url = url_base + "/" + version + "/output/pfss_latest?from=" + startdatestr + "&to=" + enddatestr
response = requests.get(request_url,  headers={"accept" : "*/*", 
                                               "apikey" : api_key })

print(response)

# success = False
# wsafilepath = ''
# conefilepath = ''
# model_time = ''
# if response.status_code == 200:

#     #Convert to json
#     js = response.json()
#     nfiles=len(js['data'])
#     #print('Found: ' + str(nfiles))
    
    
#     #get the latest file
#     i = nfiles - 1
#     found_wsa = False
#     found_cone = False

    
#     #start with the most recent file and work back in time
#     while i > 0:
#         model_time = js['data'][i]['model_run_time']
#         wsa_file_name = js['data'][i]['gong_file']
#         cone_file_name = js['data'][i]['cone_file']
        
#         wsa_file_url = url_base + "/" + version + "/" + wsa_file_name
#         cone_file_url = url_base + "/" + version + "/" + cone_file_name
        
#         if not found_wsa:
#             response_wsa = requests.get(wsa_file_url,  headers={ "apikey" : api_key })
#             if response_wsa.status_code == 200:
#                 wsafilepath = os.path.join(datadir, wsa_file_name)
#                 open(wsafilepath,"wb").write(response_wsa.content)
#                 found_wsa = True
#         if not found_cone: 
#             response_cone = requests.get(cone_file_url,  headers={ "apikey" : api_key })
#             if response_cone.status_code == 200:
#                 conefilepath = os.path.join(datadir, cone_file_name)
#                 open(conefilepath,"wb").write(response_cone.content)
#                 found_cone = True
#         i = i - 1
#         if found_wsa and found_wsa:
#             success = True
#             break
# #else: 
#     #print('Found: 0')
    