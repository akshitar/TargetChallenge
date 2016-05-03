import json
from pprint import pprint
data = None
def recObj(objStr):
    try:
        obj = json.loads(objStr)
    except:
        return ''
    result = {}
    result['guest_id'] = int(obj['guest_id'])
    items.append(int(obj['guest_id']))
    values =  obj['recs'][:5]
    while len(values) < 5:
        values.append(0)
    result['recs'] = values
    return json.dumps(result)
count = 0
with open('jayResultsFinal.json', 'a') as results:
    with open('cleanedResultsJay.json') as data_file:
        for x in data_file:
            try:
                data = json.loads(x)
                y = data['guest_id']
                y = data['recs']
                results.write(x)
            except Exception as err:
                print(count)
                count = count + 1
        data_file.close
    results.close
