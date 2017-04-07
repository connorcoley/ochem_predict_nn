# Import relevant packages
from __future__ import print_function
from global_config import USE_STEREOCHEMISTRY
import numpy as np
import cPickle as pickle
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)
import os
import sys
import time
import argparse
from ochem_predict_nn.utils.descriptors import edits_to_vectors

FROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lowe_data_edits')
if not os.path.isdir(FROOT):
	os.mkdir(FROOT)

def get_candidates(n = 10):
	'''
	Pull n example reactions, their candidates, and the true answer
	'''

	from ochem_predict_nn.utils.database import collection_candidates
	examples = collection_candidates()

	# Define generator
	class Generator():
		def __init__(self):
			self.done_ids = set()
			self.done_smiles = set()

		def get_sequential(self):
			'''Sequential'''
			for doc in examples.find({'found': True}, no_cursor_timeout = True):
				try:
					if not doc: continue 
					if doc['_id'] in self.done_ids: continue
					if doc['reactant_smiles'] in self.done_smiles: 
						print('New ID {}, but old reactant SMILES {}'.format(doc['_id'], doc['reactant_smiles']))
						continue
					self.done_ids.add(doc['_id'])
					self.done_smiles.add(doc['reactant_smiles'])
					yield doc
				except KeyboardInterrupt:
					print('Terminated early')
					quit(1)

	gen = Generator()
	generator = enumerate(gen.get_sequential())

	# Initialize (this is not the best way to do this...)
	reaction_candidate_edits = []
	reaction_candidate_smiles = []
	reaction_true_onehot = []
	reaction_true = []
	counter = 0
	for i, reaction in generator:

		candidate_smiles = [a for (a, b) in reaction['edit_candidates']]
		candidate_edits =    [b for (a, b) in reaction['edit_candidates']]
		reactant_smiles = reaction['reactant_smiles']
		product_smiles_true = reaction['product_smiles_true']

		reactants_check = Chem.MolFromSmiles(str(reactant_smiles))
		if not reactants_check:
			print('######### Could not parse reactants - that is weird...')
			print(reactant_smiles)
			continue

		bools = [product_smiles_true == x for x in candidate_smiles]
		print('rxn. {} : {} true entries out of {}'.format(i, sum(bools), len(bools)))
		if sum(bools) > 1:
			print('More than one true? Will take first one')
			pass
		if sum(bools) == 0:
			print('##### True product not found / filtered out #####')
			continue

		# Sort together and append
		zipsort = sorted(zip(bools, candidate_smiles, candidate_edits))
		zipsort = [[(y, z, x) for (y, z, x) in zipsort if y == 1][0]] + \
				  [(y, z, x) for (y, z, x) in zipsort if y == 0]

		if sum([y for (y, z, x) in zipsort]) != 1:
			print('New sum true: {}'.format(sum([y for (y, z, x) in zipsort])))
			print('## wrong number of true results?')
			raw_input('Pausing...')

		reaction_candidate_edits_compact = [
				';'.join([
					','.join(x[0]),
					','.join(x[1]),
					','.join(['%s-%s-%s' % tuple(blost) for blost in x[2]]),
					','.join(['%s-%s-%s' % tuple(bgain) for bgain in x[3]]),
				])
			for (y, z, x) in zipsort]

		# Use fingerprint length 1024
		prod_FPs = np.zeros((len(zipsort), 1024), dtype = bool)
		for i, candidate in enumerate([z for (y, z, x) in zipsort]):
			try:
				prod = Chem.MolFromSmiles(str(candidate))
				prod_FPs[i, :] = np.array(AllChem.GetMorganFingerprintAsBitVect(prod, 2, nBits = 1024), dtype = bool)
			except Exception as e:
				print(e)
				continue

		pickle.dump((
			reaction_candidate_edits_compact,
			edits_to_vectors([], reactants_check, return_atom_desc_dict = True, ORIGINAL_VERSION = True),
			prod_FPs,
			[y for (y, z, x) in zipsort],
		), fid_data, pickle.HIGHEST_PROTOCOL)

		pickle.dump((
			str(reaction['_id']),
			str(reactant_smiles) + '>>' + str(product_smiles_true) + '[{}]'.format(len(zipsort)),
			[z for (y, z, x) in zipsort],
			reaction_candidate_edits_compact,
		), fid_labels, pickle.HIGHEST_PROTOCOL)

		counter += 1


		if counter == n: break
	return reaction_candidate_edits, reaction_true_onehot, reaction_candidate_smiles, reaction_true


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--num', type = int, default = 100,
						help = 'Number of candidate sets to read, default 100')
	parser.add_argument('--tag', type = str, default = 'lowe',
						help = 'Data tag, default = lowe')
	args = parser.parse_args()

	n = int(args.num)

	legend_data = {
		'candidate_edits_compact': 0,
		'atom_desc_dict': 1,
		'prod_FPs': 2,
		'reaction_true_onehot': 3,
		'N_examples': n,
	}

	legend_labels = {
		'rxdid': 0,
		'reaction_true': 1,
		'candidate_smiles': 2,
		'candidate_edits_compact': 3,
		'N_examples': n,
	}

	tag = str(args.tag)

	if not os.path.isdir(FROOT):
		os.mkdir(FROOT)

	fid_data = open(os.path.join(FROOT, '{}_data.pickle'.format(tag)), 'wb')
	fid_labels = open(os.path.join(FROOT, '{}_labels.pickle'.format(tag)), 'wb')

	# First entry is the legend
	pickle.dump(legend_data, fid_data, pickle.HIGHEST_PROTOCOL)
	pickle.dump(legend_labels, fid_labels, pickle.HIGHEST_PROTOCOL)

	# Now get candidates
	get_candidates(n = n)
