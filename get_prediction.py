import requests
import json

def get_predictions(request):
    city = request.data.decode("utf-8")
    if city == "test":
        r_to_tf = requests.post("http://35.237.210.192:80", data=json.dumps(city))
    else:
        r_to_api = requests.post("https://us-central1-careful-time-232710.cloudfunctions.net/api_to_tf", data=city)
        print(r_to_api.content)
        data_from_api = r_to_api.content.decode("utf-8")
        r_to_tf = requests.post("http://35.237.210.192:80", data=data_from_api)
        print(r_to_tf.content)
    return r_to_tf.content