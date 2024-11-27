import requests

def get_wikidata_id(entity):
	url = "https://www.wikidata.org/w/api.php"
	params = {
		"action": "wbsearchentities",
		"language": "en",
		"format": "json",
		"search": entity
	}

	response = requests.get(url, params=params)
	if response.status_code == 200:
		data = response.json()
		if data["search"]:
			return [(d["id"], d["label"], d["description"]) for d in data["search"]]
	return None