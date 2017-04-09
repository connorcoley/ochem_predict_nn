from pymongo import MongoClient    # mongodb plugin
client = MongoClient('mongodb://username:password@server.com/authenticationDB', 27017)

def collection_example_reactions_smilesonly():
    db = client['reaction_examples']
    collection = db['lowe_1976-2013_USPTOgrants']
    return collection

def collection_example_reactions_details():
    db = client['reaction_examples']
    collection = db['lowe_1976-2013_USPTOgrants_reactions']
    return collection

def collection_templates():
    db = client['askcos_transforms']
    collection = db['lowe_refs_general_v3']
    return collection

def collection_candidates():
    db = client['prediction']
    collection = db['candidate_edits_8_9_16']
    return collection 