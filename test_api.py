import requests

url = "http://127.0.0.1:5000/predict"
#files = {'image':open('D:\face_mask_detection\dataset\with_mask\with_mask_26.jpg', 'rb')} 
 # replace 'test.jpg' with your test image path
files = {'image': open(r'D:\face_mask_detection\dataset\with_mask\with_mask_26.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())
