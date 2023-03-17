# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:39:28 2023

@author: vy902033
"""
import requests
import os

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

response3 = requests.get(request_url+"?object_name=pfss_output/windbound_b_pfss20221220.12.nc", \
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

request_url = url_base + "/" + version + "/output/pfss_output?from=" + startdatestr + "&to=" + enddatestr
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
    