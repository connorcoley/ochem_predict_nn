# Import relevant packages
from __future__ import print_function
from global_config import USE_STEREOCHEMISTRY
import numpy as np
import os
import sys
import argparse
import h5py # needed for save_weights, fails otherwise
from keras import backend as K 
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import *
import ochem_predict_nn.main.transformer as transformer 
from ochem_predict_nn.utils.canonicalization import SmilesFixer
from pymongo import MongoClient    # mongodb plugin
from ochem_predict_nn.utils.summarize_reaction_outcome import summarize_reaction_outcome
from ochem_predict_nn.utils.descriptors import edits_to_vectors, oneHotVector # for testing
from ochem_predict_nn.main.score_candidates_from_edits_compact import build
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # for visualization
import scipy.stats as ss
import itertools
import time
from tqdm import tqdm

def reactants_to_candidate_edits(reactants):
	candidate_list = [] # list of tuples of (product smiles, edits required)
	for template in tqdm(Transformer.templates):
		# Perform transformation
		try:
			outcomes = template['rxn_f'].RunReactants([reactants])
		except Exception as e:
			if v: print(e)
			continue
		if not outcomes: continue # no match
		for j, outcome in enumerate(outcomes):
			outcome = outcome[0] # all products represented as single mol by transforms
			try:
				outcome.UpdatePropertyCache()
				Chem.SanitizeMol(outcome)
				[a.SetProp('molAtomMapNumber', a.GetProp('old_molAtomMapNumber')) \
					for (i, a) in enumerate(outcome.GetAtoms()) \
					if 'old_molAtomMapNumber' in a.GetPropsAsDict()]
			except Exception as e:
				if v: print(e)
				continue
			if v: print('Outcome SMILES: {}'.format(Chem.MolToSmiles(outcome)))

			# Reduce to largest (longest) product only
			candidate_smiles = Chem.MolToSmiles(outcome, isomericSmiles = USE_STEREOCHEMISTRY)
			candidate_smiles = max(candidate_smiles.split('.'), key = len)
			outcome = Chem.MolFromSmiles(candidate_smiles)
				
			# Find what edits were made
			try:
				edits = summarize_reaction_outcome(reactants, outcome)
			except KeyError as e:
				print('Do you have the custom RDKit version installed? Maybe not...')
				raise(e)
			if v: print(edits)

			# Remove mapping before matching
			[x.ClearProp('molAtomMapNumber') for x in outcome.GetAtoms() if x.HasProp('molAtomMapNumber')] # remove atom mapping from outcome

			# Overwrite candidate_smiles without atom mapping numbers
			candidate_smiles = Chem.MolToSmiles(outcome, isomericSmiles = USE_STEREOCHEMISTRY)

			# Add to ongoing list
			if (candidate_smiles, edits) not in candidate_list:
				candidate_list.append((candidate_smiles, edits))

	return candidate_list 

def preprocess_candidate_edits(reactants, candidate_list):
	candidate_smiles = [a for (a, b) in candidate_list]
	candidate_edits =  [b for (a, b) in candidate_list]

	print('Generated {} unique edit sets'.format(len(candidate_list)))
	padUpTo = len(candidate_list)
	N_e1 = 20
	N_e2 = 20
	N_e3 = 20
	N_e4 = 20

	N_e1_trim = 1
	N_e2_trim = 1
	N_e3_trim = 1
	N_e4_trim = 1

	# Initialize
	x_h_lost = np.zeros((1, padUpTo, N_e1, F_atom))
	x_h_gain = np.zeros((1, padUpTo, N_e2, F_atom))
	x_bond_lost = np.zeros((1, padUpTo, N_e3, F_bond))
	x_bond_gain = np.zeros((1, padUpTo, N_e4, F_bond))
	x = np.zeros((1, padUpTo, 1024))

	# Get reactant descriptors
	atom_desc_dict = edits_to_vectors([], reactants, return_atom_desc_dict = True)

	# Populate arrays
	for (c, edits) in enumerate(candidate_edits):
		if c == padUpTo: break
		edit_h_lost_vec, edit_h_gain_vec, \
			edit_bond_lost_vec, edit_bond_gain_vec = edits_to_vectors(edits, reactants, atom_desc_dict = atom_desc_dict)

		N_e1_trim = max(N_e1_trim, len(edit_h_lost_vec))
		N_e2_trim = max(N_e2_trim, len(edit_h_gain_vec))
		N_e3_trim = max(N_e3_trim, len(edit_bond_lost_vec))
		N_e4_trim = max(N_e4_trim, len(edit_bond_gain_vec))

		for (e, edit_h_lost) in enumerate(edit_h_lost_vec):
			if e >= N_e1: continue
			x_h_lost[0, c, e, :] = edit_h_lost
		for (e, edit_h_gain) in enumerate(edit_h_gain_vec):
			if e >= N_e2: continue
			x_h_gain[0, c, e, :] = edit_h_gain
		for (e, edit_bond_lost) in enumerate(edit_bond_lost_vec):
			if e >= N_e3: continue
			x_bond_lost[0, c, e, :] = edit_bond_lost
		for (e, edit_bond_gain) in enumerate(edit_bond_gain_vec):
			if e >= N_e4: continue
			x_bond_gain[0, c, e, :] = edit_bond_gain

		if BASELINE_MODEL or HYBRID_MODEL:
			prod = Chem.MolFromSmiles(str(candidate_smiles[c]))
			if prod is not None:
				x[0, c, :] = np.array(AllChem.GetMorganFingerprintAsBitVect(prod, 2, nBits = 1024), dtype = bool)

	# Trim down
	x_h_lost = x_h_lost[:, :, :N_e1_trim, :]
	x_h_gain = x_h_gain[:, :, :N_e2_trim, :]
	x_bond_lost = x_bond_lost[:, :, :N_e3_trim, :]
	x_bond_gain = x_bond_gain[:, :, :N_e4_trim, :]

	# Get rid of NaNs
	x_h_lost[np.isnan(x_h_lost)] = 0.0
	x_h_gain[np.isnan(x_h_gain)] = 0.0
	x_bond_lost[np.isnan(x_bond_lost)] = 0.0
	x_bond_gain[np.isnan(x_bond_gain)] = 0.0
	x_h_lost[np.isinf(x_h_lost)] = 0.0
	x_h_gain[np.isinf(x_h_gain)] = 0.0
	x_bond_lost[np.isinf(x_bond_lost)] = 0.0
	x_bond_gain[np.isinf(x_bond_gain)] = 0.0

	if BASELINE_MODEL:
		return [x] 
	elif HYBRID_MODEL:
		return [x_h_lost, x_h_gain, x_bond_lost, x_bond_gain, x]

	return [x_h_lost, x_h_gain, x_bond_lost, x_bond_gain]

def score_candidates(reactants, candidate_list, xs):

	pred = model.predict(xs, batch_size = 20)[0]
	rank = ss.rankdata(pred)

	fname = raw_input('Enter file name to save to: ') + '.dat'
	with open(os.path.join(FROOT, fname), 'w') as fid:
		fid.write('FOR REACTANTS {}\n'.format(Chem.MolToSmiles(reactants)))
		fid.write('Candidate product\tCandidate edit\tProbability\tRank\n')
		for (c, candidate) in enumerate(candidate_list):
			candidate_smile = candidate[0]
			candidate_edit = candidate[1]
			fid.write('{}\t{}\t{}\t{}\n'.format(
				candidate_smile, candidate_edit, pred[c], 1 + len(pred) - rank[c]
			))
	print('Wrote to file {}'.format(os.path.join(FROOT, fname)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--tag', type = str,
						help = 'Tag for model to load from')
	parser.add_argument('--mincount', type = int, default = 50,
						help = 'Mincount of templates, default 50')
	parser.add_argument('-v', type = int, default = 1,
						help = 'Verbose? default 1')
	args = parser.parse_args()

	v = bool(int(args.v))
	FROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		'main', 'output', str(args.tag))

	MODEL_FPATH = os.path.join(FROOT, 'model.json')
	WEIGHTS_FPATH = os.path.join(FROOT, 'weights.h5')
	ARGS_FPATH = os.path.join(FROOT, 'args.json')

	mol = Chem.MolFromSmiles('[C:1][C:2]')
	(a, _, b, _) = edits_to_vectors((['1'],[],[('1','2',1.0)],[]), mol)
	F_atom = len(a[0])
	F_bond = len(b[0])

	# Silence warnings
	from rdkit import RDLogger
	lg = RDLogger.logger()
	lg.setLevel(4)

	# Load transformer
	from ochem_predict_nn.utils.database import collection_templates
	templates = collection_templates()
	Transformer = transformer.Transformer()
	Transformer.load(templates, mincount = int(args.mincount), get_retro = False, get_synth = True, lowe = True)
	print('Out of {} database templates,'.format(templates.count()))
	print('Loaded {} templates'.format(Transformer.num_templates))
	Transformer.reorder()
	print('Sorted by count, descending')

	# Load models - must rebuild...
	with open(ARGS_FPATH, 'r') as fid:
		import json
		args_dict = json.load(fid)
		print('Arguments loaded from saved model:')
		print(args_dict)

	HYBRID_MODEL = bool(int(args_dict['hybrid']))
	BASELINE_MODEL = bool(int(args_dict['baseline']))

	try: # usually fails because of custom Theano usage
		model = model_from_json(open(MODEL_FPATH).read())
	except NameError:

		model = build(F_atom = F_atom, F_bond = F_bond, N_h1 = int(args_dict['Nh1']), 
			N_h2 = int(args_dict['Nh2']), N_h3 = int(args_dict['Nh3']), N_hf = int(args_dict['Nhf']), 
			l2v = float(args_dict['l2']), lr = 0.0, optimizer = args_dict['optimizer'], 
			inner_act = args_dict['inner_act'], HYBRID_MODEL = HYBRID_MODEL, 
			BASELINE_MODEL = BASELINE_MODEL,
		)
	model.compile(
		loss = 'categorical_crossentropy',
		optimizer = SGD(lr = 0.0),
		metrics = ['accuracy']
	)
	model.load_weights(WEIGHTS_FPATH)

	# Wait for user prompt
	while True:
		reactants = Chem.MolFromSmiles(raw_input('Enter SMILES of reactants: '))
		if not reactants:
			print('Could not parse!')
			continue

		print('Number of reactant atoms: {}'.format(len(reactants.GetAtoms())))
		# Report current reactant SMILES string
		print('Reactants w/o map: {}'.format(Chem.MolToSmiles(reactants)))
		# Add new atom map numbers
		[a.SetProp('molAtomMapNumber', str(i+1)) for (i, a) in enumerate(reactants.GetAtoms())]
		# Report new reactant SMILES string
		print('Reactants w/ map: {}'.format(Chem.MolToSmiles(reactants)))

		# Generate candidates
		candidate_list = reactants_to_candidate_edits(reactants)
		# Convert to matrices
		xs = preprocess_candidate_edits(reactants, candidate_list)
		# Score and save to file
		score_candidates(reactants, candidate_list, xs)
