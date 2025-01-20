import requests

response = requests.post("http://35.239.75.232:9090/mia", files={"file": open("submission.csv", "rb")}, headers={"token": "15184357"})
