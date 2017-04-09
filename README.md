# ochem_predict_nn

## Project summary

This project uses open source reaction data from the USPTO (pre-extracted by Daniel Lowe, https://bitbucket.org/dan2097/patent-reaction-extraction/downloads) to train a neural network model to predict the outcomes of organic reactions. Reaction templates are used to enumerate potential products; a neural network scores each product and ranks likely outcomes. By examining thousands of experimental outcomes, the model learns which modes of reactivity are likely to occur. The full details can be found at TBD.

The code relies on Keras for its machine learning components using the Theano background. RDKit is used for all chemistry-related parsing and processing. Please note that due to the unique reaction representation used, generating candidate outcomes requires the modified RDKit version available at https://github.com/connorcoley/rdkit. In the modified version, atom-mapping numbers associated with reactant molecules are preseved after calling ```RunReactants```. The code is set up to use MongoDB to store reaction examples, transform strings, and candidate sets. A mongodump containing all data used in the project can be found at https://figshare.com/account/articles/4833482. The database/collection names are defined in ```utils/database.py```.

## Generating templates

Reaction templates are extracted from ca. 1M atom-mapped reaction SMILES strings using ```data/generate_reaction_templates.py```. They are designed to be overgeneral to cover a broad range of chemistry at the expense of specificity. The extracted templates can be found in the mongodump, so they do not need to be re-extracted.

## Generating candidates

A forward enumeration algorithm is used to generate plausible candidates for each set of reactants using ```data/generate_candidates_edits_fullgrants.py``` with the help of the ```main/transformer.py``` class. Reagents, catalysts, and solvents (if present) are allowed to react in addition to the reactants. This makes the prediction task artificially hard (as the reaction database already contains information about which atoms react), but it is reasonable given that role labelling was performed with knowledge of the reaction outcome. Candidates are inserted into a MongoDB automatically. 

## Preprocessing candidates

To prepare the data for training, ```data/preprocess_candidate_edits_compact.py``` is used to generate necessary atom-level descriptors for reactant molecules, which will be used in the edit-based representation. Data is pickled in a compressed format to minimize storage size and file read limitations, but is expanded during training and testing into its full many-tensor representation.

## Model training/testing

Models are trained and tested using ```main/score_candidates_from_edits_compact.py```. Many command-line options are available to set different architecture/training parameters, including which fold of a 5-fold CV is being run. A demo model using just 10 reactions is included in ```main/output/10rxn_demo1```.

## Trained model testing

An already-trained model can be loaded using ```scripts/lowe_interactive_predict.py``` to make predictions on demand. You will be prompted to enter reactant SMILES strings; the results of the forward prediction are saved as a table of products, scores, and probabilities.
