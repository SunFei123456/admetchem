import json
import sys

from AModel import AModel

smiles = ["CC(=O)Oc1ccc(/C=C/C(=O)OCCOC(=O)/C=C/c2ccc(OC(C)=O)c(OC(C)=O)c2)cc1OC(C)=O",
"NCC(=O)O",
"COC(=O)C(COC(=O)/C=C/c1ccc(OC(C)=O)c(OC(C)=O)c1)NC(=O)/C=C/c1ccc(OC(C)=O)c(OC(C)=O)c1",
"CCC(C)c1ccccc1O"]
config_path = sys.argv[1]
config = json.load(open(config_path, 'r'))
data_args = config['data']
train_args = config['train']
train_args['data_name'] = config_path.split('/')[-1].strip('.json')
model_args = config['model']
model_args['device'] = train_args['device']
model_args['dropout'] = 0.1
amodel = AModel(model_args).to(model_args['device'])

print(amodel(smiles))