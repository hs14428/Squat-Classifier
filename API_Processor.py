import requests as rq
import json
from datetime import datetime

BASE_URL = 'http://hs14428.pythonanywhere.com'

payload = {'input': 'Successful input'}
response = rq.get(BASE_URL)#, params=payload)

json_values = response.json()

print(json_values)
print(json_values['1'])

# rq_input = json_values['input']
# timestamp = json_values['timestamp']
# return_message = json_values['return_message']
#
# print(f'input: {rq_input}')
# print(f'timestamp: {datetime.fromtimestamp(timestamp)}')
# print(f'return_message: {return_message}')




