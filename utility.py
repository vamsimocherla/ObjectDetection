import json

json_file_path = '/home/vamsimocherla/PycharmProjects/dl4od/datasets/home_objects_train_attributes.json'
with open(json_file_path) as json_file:
    data = json.load(json_file)
    ind = 0
    for key, value in data['region']['type']['options'].items():
        # print('{} {}'.format(p['supercategory'], p['name']))
        print("{}:{},".format(key, value))

    d = data['region']['type']['options']
    print(d['person'])
