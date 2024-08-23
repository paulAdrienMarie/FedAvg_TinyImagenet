import json

with open("dataset_infos.json") as f:
    ds_info = json.loads(f.read())
    
with open("classes.json") as f:
    classes = json.loads(f.read())
    
res = {id_: classe for id_, classe in classes.items() if id_ in ds_info["Maysee--tiny-imagenet"]["features"]["label"]["names"]}

res_ = {
    "id2label": {},
    "label2id": {}
    }

items = list(res.values())

for j in range(len(res.items())):
    res_["id2label"][j] = items[j]
    res_["label2id"][items[j]] = j
    
    
with open("config.json","w") as f:
    json.dump(res_,f)