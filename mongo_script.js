//create the ModelHub database and connect to it
var db = connect('127.0.0.1:27017/ModelHubDB'),
    ModelHubDB = null;

print(' ** Database ModelHubDB Created **');

//create the dlmodels collection and add documents to it
db.dlmodels.insert({'ModelName': 'Dummy', 'ModelType': 'FFN', 'ModelContent': null});

print(' ** Document Archetype for a Deep Learning Model Created ** ');

