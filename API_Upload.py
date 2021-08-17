import requests as rq

TOKEN = '3de90533da13f409953870b54506da46df848224'
FILEPATH = 'files/path/home/hs14428/mysite'
API = 'https://www.pythonanywhere.com/api/v0/user/hs14428/'
PANAME = '/test.mp4'
MP4_FILEPATH = 'test.mp4'

files = {'content': open(MP4_FILEPATH, 'rb')}

res = rq.post(API + FILEPATH + PANAME,
              files=files,
              headers={'Authorization': 'Token ' + TOKEN})

print(res)
