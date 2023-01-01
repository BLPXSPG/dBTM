/* 
   Copyright (C) 2011 Knowledge Media Institute, The Open University. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
	 you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Chenghua Lin, c.lin@open.ac.uk 
           Revised by Yulan He, 14/05/2011
*/
   
   

#include "model.h"


using namespace std;


model::model(void)
{

	wordmapfile = "wordmap.txt";
	tassign_suffix = ".tassign";
	pi_suffix = ".pi";
	theta_suffix = ".theta";
	phi_suffix = ".phi";
	others_suffix = ".others";
	twords_suffix = ".twords";
	opt_mu_mode = "decay";
	//results_suffix = ".results";

	docLabFile = "";
	
	// epoch t specific dataset parameters
	numScales = 1;
	maxSlices = 0;
	mode = MODE_NONE;
	
	numTopics = 50;
	numSentiLabs = 3;
	vocabSize = 0;
	numDocs = 0;
	epochSize = 0;
	aveDocLength = 0;
	
	niters = 1000;
	liter = 0;
	savestep = 200; 
	twords = 20; 
	updateParaStep = 40;
	epochID = 0;
	upsilon = 1.0;
	time_duration = 0.0;

	_alpha  = -1.0;
	_beta = -1.0;
	_gamma = -1.0;

	useHistoricalAlpha = false;

	putils = new utils();
}

model::~model(void)
{
	if (putils)
		delete putils;
}

int model::init(int argc, char ** argv) 
{
	// call parse_args
  if (putils->parse_args(argc, argv, this))
		return 1;
    
	cout<<"data_dir = "<<data_dir<<endl;
	cout<<"result_dir = "<<result_dir<<endl;
	cout<<"datasetFile = "<<datasetFile<<endl;
	cout<<"sentiLexFile = "<<sentiLexFile<<endl;
	cout<<"docLabFile = "<<docLabFile<<endl;
	cout<<"wordmapfile = "<<wordmapfile<<endl;
	//cout<<"model_status = "<<model_status<<endl;
	//cout<<"epochLength = "<<epochLength<<endl;
	cout<<"numScales = "<<numScales<<endl;
	cout<<"numTopics = "<<numTopics<<endl;
	cout<<"numSentiLabs = "<<numSentiLabs<<endl;
	cout<<"niters = "<<niters<<endl;
	cout<<"savestep = "<<savestep<<endl;
	cout<<"twords = "<<twords<<endl;
	cout<<"updateParaStep = "<<updateParaStep<<endl;
	cout<<"opt_mu_mode = "<<opt_mu_mode<<endl;
	cout<<"_alpha = "<<_alpha<<endl;
	cout<<"_beta = "<<_beta<<endl;
	cout<<"_gamma = "<<_gamma<<endl;
	cout<<"useHistoricalAlpha = "<<useHistoricalAlpha<<endl;
	
	return 0;
}

int model::excute_model()
{
	clock_t start; 
	clock_t finish;
	ifstream fin; // global file handle
	epochID = 0;
	string inputFile;
	
	stringstream ss;
	  
	while (1) {
		ss.str(""); ss.clear();
		//ss << data_dir << datasetFile << "_epoch" << epochID << ".dat";
		ss << data_dir << datasetFile << epochID << ".dat";
		inputFile = ss.str();
		
		fin.open(inputFile.c_str(), ifstream::in);
	
		if(!fin) {
			printf("Unable to read %s. Finish processing files.\n", inputFile.c_str());
			return 0; 
		}

		// start counting the processing time
		time_duration = 0.0;
		start = clock();	

		pdataset = new dataset(result_dir, epochID, corpusVocab, sentiLex); 
		if (epochID == 0) {
			if (sentiLexFile != "") {
				if (pdataset->read_senti_lexicon((sentiLexFile).c_str())) {
					printf("Error! Can not open sentiFile %s to read!\n", (sentiLexFile).c_str());
					delete pdataset;
					return 1;  // read sentiLex fail!
				}
			
				sentiLex = pdataset->sentiLex; 
			}
		
			if (docLabFile != "") {
				if (read_doc_labels(docLabFile.c_str())) {
					printf("Error! Can not open doc label file %s to read!\n", (docLabFile).c_str());
					return 1;  // read doc label file fail!
				}
			}
		}
							
		if(pdataset->read_dataStream(fin)) {
			printf("Error in calling function read_dataStream()! \n");
			delete pdataset;
			return 1;
		}

		//pdataset->print_epoch_statistics();
		word2atr = pdataset->word2atr;
		id2word =  pdataset->id2word;
		newWords = pdataset->newWords;			    
		corpusVocabSize = pdataset->corpusVocabSize;
		corpusVocab = pdataset->corpusVocab;   // <string, int> ==> <word, corpusWideVocabID>
		epochVocabID2corpusVocabID = pdataset->epochVocabID2corpusVocabID;
		corpusVocabID2epochVocabID = pdataset->corpusVocabID2epochVocabID;

		if (epochID == 0) {
			init_model_parameters();
			
			if (mode != MODE_NONE)
				init_dynamic_mode_parameters();
		}
		else {
			if (reset_model_parameters()) return 1;

			if (mode != MODE_NONE) { // account for historical epochs
				if (set_ntlzw_epsilon()) return 1;
				if (reset_dynamic_mode_parameters()) return 1;
			}
			else { // don't account for historical epochs
				if (prior2beta()) {
					delete pdataset;
					return 1;
				}
			}
			//newWords_prior2beta();  // incorporate word prior info
		}
		
		if (init_estimate()) return 1;
		if(estimate()) return 1;

		// end of processing time
		finish = clock();
		time_duration = (double)(finish - start) / CLOCKS_PER_SEC;
		if (save_model_time_duration(result_dir + "timeLog.txt")) {
			delete pdataset;
			return 1;
		}

		epochID++;
	
		delete_model_parameters();

		fin.close();
	}
	
	return 0;
}


int model::set_ntlzw_epsilon()
{
	if (newWords.size() > 0) {

		// expand the dimension of epsilon_slzw 
		for (int t = 0; t < numScales; t++) {
			for (int l = 0; l < numSentiLabs; l++) {
				for (int z = 0; z < numTopics; z++) {
					for (int r = 0; r < (int)newWords.size(); r++)
						epsilon_slzw[t][l][z].push_back(0.0);
				}
			}
		}
		
		// expand the dimension of _ntlzw
		for (int t = 0; t < maxSlices; t++) {
			for (int l = 0; l < numSentiLabs; l++) {
				for (int z = 0; z < numTopics; z++) {
					for (int r = 0; r < (int)newWords.size(); r++)
						_ntlzw[t][l][z].push_back(0.0);
				}
			}
		}
	}
	
	// get word polarity prior for current epoch
	mapword2atr::iterator wordIt;
	mapword2prior::iterator sentiIt;
	
	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		wordIt = word2atr.find(sentiIt->first);
		if (wordIt != word2atr.end()) {
			for (int j = 0; j < numSentiLabs; j++)  {
				lambda_lw[j][wordIt->second.id] = sentiIt->second.labDist[j];
			}
		}
	}
	
	/*mapword2atr::iterator itatr;
		
	for (itatr = word2atr.begin(); itatr != word2atr.end(); itatr++) {
		if ((itatr->second.polarity > -1)) { 
			for (int j = 0; j < numSentiLabs; j++)  {
				// retain value '1' --> if() lambda_lw[j][it->second.id] = 0.9
				if (j == itatr->second.polarity) lambda_lw[j][itatr->second.id] = 0.9;  
				// set 0 to the corresponding vocabulary --> else ambda_lw[j][it->second.id] = 0.05
				else lambda_lw[j][itatr->second.id] = 0.05;  
			}                                          
		}
	}*/
	
	// incorporate word polarity prior (at 0th scale)
	map<int, int>::iterator it;		
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			for (int r = 0; r < corpusVocabSize; r++) {
				it = corpusVocabID2epochVocabID.find(r);
				if (it != corpusVocabID2epochVocabID.end())
					epsilon_slzw[0][l][z][r] = 0.01 * lambda_lw[l][it->second];
				else
					epsilon_slzw[0][l][z][r] = 0.01;
				//epsilon_slzw[0][l][z][r] = (double)1.0/corpusVocabSize;
			}
		}
	}

	return 0;
}

int model::init_dynamic_mode_parameters()
{
	numDocs = pdataset->numDocs;
	vocabSize = pdataset->vocabSize;
	epochSize = pdataset->epochSize;
	aveDocLength = pdataset->aveDocLength;
	corpusVocabSize = pdataset->corpusVocabSize;
	
	// maxSlices = maxSlices<<(numScales-1);
	if (mode == MODE_SKIP || mode == MODE_MULTISCALE)
		maxSlices = pow((double)2, (numScales-2));
	else if (mode == MODE_SLIDING)
		maxSlices = numScales-1;
	
	alpha_lz_old.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
		alpha_lz_old[l].resize(numTopics);

	// expected counts 
	_nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)	{
		_nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			_nlzw[l][z].resize(corpusVocabSize);
			for (int r = 0; r < corpusVocabSize; r++) {
			    _nlzw[l][z][r] = 0.0;
			}
		}
	}

	_n_slzw.resize(numScales);
	for (int s = 0; s < numScales; s++) {    
		_n_slzw[s].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			_n_slzw[s][l].resize(numTopics);
				for (int z = 0; z < numTopics; z++) {
					_n_slzw[s][l][z].resize(corpusVocabSize);
					for (int r = 0; r < corpusVocabSize; r++)
					_n_slzw[s][l][z][r] = 0.0;
				}
		}
	}
	
	_n_slz.resize(numScales);
	for (int s = 0; s < numScales; s++) {
		_n_slz[s].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			_n_slz[s][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++)
			    _n_slz[s][l][z] = 0.0;
		}
	}

	if (opt_mu_mode == "decay") {  
		mu_s.resize(numScales);
		set_mu_decayFunction();
	}
	
	else if (opt_mu_mode == "EM") {
		mu_slz.resize(numScales); // weights for epochs from [t-numScales, t-1]; 
		for (int s = 0; s < numScales; s++) {
			mu_slz[s].resize(numSentiLabs);
			for (int l = 0; l < numSentiLabs; l++) {
				mu_slz[s][l].resize(numTopics);
				for (int z = 0; z < numTopics; z++)
						mu_slz[s][l][z] = (double)1/numScales;
			}
		}
		
		mu_lz.resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			mu_lz[l].resize(numTopics);
			for (int z = 0; z < numTopics; z++) {
			    mu_lz[l][z] = 0.0;  // reset mu_lz[l][z] to 0
			    for (int s = 0; s < numScales; s++) 
	        		mu_lz[l][z] += mu_slz[s][l][z];
			}
		}
		
		opt_mu_slz.resize(numScales);
		for (int s = 0; s < numScales; s++) {
			opt_mu_slz[s].resize(numSentiLabs);
			for (int l = 0; l < numSentiLabs; l++)
				opt_mu_slz[s][l].resize(numTopics);
		}
	}
	
	epsilon_slzw.resize(numScales); 
	for (int s = 0; s < numScales; s++) {
		epsilon_slzw[s].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			epsilon_slzw[s][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++) {
				epsilon_slzw[s][l][z].resize(corpusVocabSize);
				for (int r = 0; r < corpusVocabSize; r++) {
                                  if (s == 0)
                                      epsilon_slzw[s][l][z][r] = beta_lzw[l][z][r];  // set epsilion_slzw at epoch 0 with prior information 
                                  else 
                                      epsilon_slzw[s][l][z][r] = 0.0;
				}

			}
		}
	}
			
	_ntlzw.resize(maxSlices);  // maxSlices -> numScales 
	for (int t = 0; t < maxSlices; t++) {
		_ntlzw[t].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			_ntlzw[t][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++)
				_ntlzw[t][l][z].resize(corpusVocabSize);
		}
	}	

	return 0;
}


int model::init_model_parameters()
{
	numDocs = pdataset->numDocs;
	vocabSize = pdataset->vocabSize;
	epochSize = pdataset->epochSize;
	aveDocLength = pdataset->aveDocLength;
	corpusVocabSize = pdataset->corpusVocabSize;
	
	// model real counts
	nd.resize(numDocs);

	ndl.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
		    ndl[m][l] = 0;
	}

	ndlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++)
				ndlz[m][l][z] = 0; 
		}
	}

	nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++)
			    nlzw[l][z][r] = 0;
		}
	}

	nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
		    nlz[l][z] = 0;
	}

	// posterior P
	p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
		p[l].resize(numTopics);

	// model parameters
	pi_dl.resize(numDocs);
	for (int m = 0; m < numDocs; m++)
		pi_dl[m].resize(numSentiLabs);

	theta_dlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		theta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
			theta_dlz[m][l].resize(numTopics);
	}

	phi_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		phi_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
			phi_lzw[l][z].resize(vocabSize);
	}

	// set initial hyperparameters
	//alpha
	alpha_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
		alpha_lz[l].resize(numTopics);

	alphaSum_l.resize(numSentiLabs);
	
	if (_alpha <= 0)
		_alpha =  (double)aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);

	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
	    for (int z = 0; z < numTopics; z++) {
		    alpha_lz[l][z] = _alpha;
		    alphaSum_l[l] += alpha_lz[l][z];
	    }
	}

	//beta
	if (_beta <= 0) _beta = 0.01;

	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) 
		    beta_lzw[l][z][r] = _beta; 
		} 		
	}


	// word prior transformation matrix
	lambda_lw.resize(numSentiLabs); 
	for (int l = 0; l < numSentiLabs; l++) {
	  lambda_lw[l].resize(vocabSize);
		for (int r = 0; r < vocabSize; r++)
			lambda_lw[l][r] = 1; 	
	}
	// MUST init beta_lzw first before incorporating prior information into beta
	prior2beta(); 
		
	set_gamma();

	// parameters for updating \alpha and \mu 
	opt_alpha_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
		opt_alpha_lz[l].resize(numTopics);

	return 0;
}


int model::set_gamma()
{
	mapname2labs::iterator it;

	//gamma
	if (_gamma <= 0 )
		_gamma = (double)aveDocLength * 0.05 / (double)numSentiLabs;

	gamma_dl.resize(numDocs);
	gammaSum_d.resize(numDocs);
	for (int d = 0; d < numDocs; d++) {
		gamma_dl[d].resize(numSentiLabs);
		gammaSum_d[d] = 0.0;

		it = docLabs.find(pdataset->pdocs[d]->docID);
		if (it != docLabs.end()) {
			for (int l = 0; l < numSentiLabs; l++) {
				gamma_dl[d][l] = it->second[l] * _gamma * numSentiLabs;
				//gamma_dl[d][l] = it->second[l] + _gamma;
				gammaSum_d[d] += gamma_dl[d][l];
			}
		}
		else {
			for (int l = 0; l < numSentiLabs; l++) { 
				gamma_dl[d][l] = _gamma; 
				gammaSum_d[d] += gamma_dl[d][l];
			}
		}
	}

	return 0;
}

int model::reset_model_parameters()
{
	numDocs = pdataset->numDocs;
	vocabSize = pdataset->vocabSize;
	epochSize = pdataset->epochSize;
	aveDocLength = pdataset->aveDocLength;
	corpusVocabSize = pdataset->corpusVocabSize;

	// model real counts
	nd.resize(numDocs);

	ndl.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
		    ndl[m][l] = 0;
	}

	ndlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++)
				ndlz[m][l][z] = 0; 
		}
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++)
			    nlzw[l][z][r] = 0;
		}
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++)
		    nlz[l][z] = 0;
	}

	// model parameters
	pi_dl.resize(numDocs);
	for (int m = 0; m < numDocs; m++)
		pi_dl[m].resize(numSentiLabs);

	theta_dlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		theta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
			theta_dlz[m][l].resize(numTopics);
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++)
			phi_lzw[l][z].resize(vocabSize);
	}

	// hyperparameters
	if (_alpha <= 0)
		_alpha =  (double)aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);

	// reset alpha if not considering historical epochs
	if (!useHistoricalAlpha) {
		for (int l = 0; l < numSentiLabs; l++) {
			alphaSum_l[l] = 0.0;
		    for (int z = 0; z < numTopics; z++) {
			    alpha_lz[l][z] = _alpha;
			    alphaSum_l[l] += alpha_lz[l][z];
		    }
		}
	}
	
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) {
				beta_lzw[l][z][r] = 0.01;		// don't account for historical epochs
			}
		}
	}
	
	set_gamma();

	// word prior transformation matrix
	for (int l = 0; l < numSentiLabs; l++) {
	  lambda_lw[l].resize(vocabSize);
		for (int r = 0; r < vocabSize; r++)
			lambda_lw[l][r] = 1; 	
	}
	
	return 0;
}

int model::reset_dynamic_mode_parameters()
{
	numDocs = pdataset->numDocs;
	vocabSize = pdataset->vocabSize;
	epochSize = pdataset->epochSize;
	aveDocLength = pdataset->aveDocLength;
	corpusVocabSize = pdataset->corpusVocabSize;

	// expected counts 
	for (int l = 0; l < numSentiLabs; l++)	{
		for (int z = 0; z < numTopics; z++) {
			_nlzw[l][z].resize(corpusVocabSize);
			for (int r = 0; r < corpusVocabSize; r++) {
			    _nlzw[l][z][r] = 0.0;
			}
		}
	}

	for (int s = 0; s < numScales; s++) {    
    for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++) {
				_n_slzw[s][l][z].resize(corpusVocabSize);
				_n_slz[s][l][z] = 0.0;
				for (int r = 0; r < corpusVocabSize; r++)
			    _n_slzw[s][l][z][r] = 0.0;
			}
    }
	}

	// hyperparameters
	if (set_dynamic_hyperparameters()) 
		return 1;
	
	return 0;
}

int model::set_dynamic_hyperparameters()
{
	// init alpha_lz with alpha_lz_old; i.e., assume that the inital value of '\alpha' is equal to '\alpha_t-1'
	if (mode != MODE_NONE && useHistoricalAlpha) {
		for (int l = 0; l < numSentiLabs; l++) {
			alphaSum_l[l] = 0.0;
			for (int t = 0; t < numTopics; t++) {
				alpha_lz[l][t] = alpha_lz_old[l][t];
				alphaSum_l[l] += alpha_lz[l][t];
			}
		}
	}

	if (opt_mu_mode == "EM")  {
		for (int s = 0; s < numScales; s++) {
			for (int l = 0; l < numSentiLabs; l++) {
				for (int z = 0; z < numTopics; z++) {
						mu_slz[s][l][z] = (double)1/(numScales);
				}
			}
		}
	}

	// in order to calculate \beta, mu_slz and \epsilon_slzw have to be updated in advanced in init_parameters()
	if (update_beta()) return 1;

	return 0;
}


int model::prior2beta()
{
	mapword2atr::iterator wordIt;
	mapword2prior::iterator sentiIt;
	
	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		wordIt = word2atr.find(sentiIt->first);
		if (wordIt != word2atr.end()) {
			for (int j = 0; j < numSentiLabs; j++)  {
				lambda_lw[j][wordIt->second.id] = sentiIt->second.labDist[j];
			}
		}
	}
	
	// Note: the 'r' index of lambda[j][r] is corresponding to the vocabulary ID. 
	// Therefore the correct prior info can be incorporated to corresponding word cound nlzw,
	// as 'w' is also corresponding to the vocabulary ID. 
  for (int l = 0; l < numSentiLabs; l++) {                                          
		for (int z = 0; z < numTopics; z++) {
			betaSum_lz[l][z] = 0.0;
		    for (int r = 0; r < vocabSize; r++) {
			    beta_lzw[l][z][r] = beta_lzw[l][z][r] * lambda_lw[l][r];  
			    betaSum_lz[l][z] += beta_lzw[l][z][r];
		    }
		}
	}

	return 0;
}

int model::update_beta()
{   
	map<int, int>::iterator it;
		
	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			betaSum_lz[l][z] = 0.0;
			for (int r = 0; r < vocabSize; r++) {
				
				it = epochVocabID2corpusVocabID.find(r);
				if (it != epochVocabID2corpusVocabID.end()) {
					beta_lzw[l][z][r] = 0.0;
					for (int s = 0; s < numScales; s++)  { // (numScales+1) to include the special case scale = 0;
						if (opt_mu_mode == "decay")
							beta_lzw[l][z][r] += mu_s[s] * epsilon_slzw[s][l][z][it->second];
						else
							beta_lzw[l][z][r] += mu_slz[s][l][z] * epsilon_slzw[s][l][z][it->second];
					  	
						betaSum_lz[l][z] += beta_lzw[l][z][r];
					}
				}
				else {
					printf("Error! update_beta: Can't find word [%d] in corpus vocab list!\n", r);
					return 1;
				}
			}
		}
	}
	
	
  return 0; 
}


void model::compute_phi_lzw()
{
	for (int l = 0; l < numSentiLabs; l++)  {
	    for (int z = 0; z < numTopics; z++) {
			for(int r = 0; r < vocabSize; r++)
			     phi_lzw[l][z][r] = (nlzw[l][z][r] + beta_lzw[l][z][r]) / (nlz[l][z] + betaSum_lz[l][z]);
		}
	}

}


// ONLY call this function after finishing all the Gibbs sampling. NOTE: we have to account for epsilon_(0) for all scales
int model::compute_epsilon_slzw()
{
	for (int s = 1; s < numScales; s++) {
		for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++) {
				for (int r = 0; r < corpusVocabSize; r++) {
					// s = 0 is for current epoch, value will be set at the expand function
					// Redo nomalization: need to take care of the case when _n_slz[s][l][t] == 0;  !!!
					epsilon_slzw[s][l][z][r] = (double) (_n_slzw[s][l][z][r] + 1) / (_n_slz[s][l][z] + corpusVocabSize);
				}	
			}
		}
	}
	
	return 0;
}


// NEED to initialise _n_slzw and _n_slz
int model::compute_expected_counts()
{
	int s, t, l, z, r, idx;
	map<int, int>::iterator it;
	double tmp;
	
	// compute_expected _nlzw
	for (l = 0; l < numSentiLabs; l++) {
		for (z = 0; z < numTopics; z++) {
		    for (r = 0; r < corpusVocabSize; r++) {
		    	it = corpusVocabID2epochVocabID.find(r);
				if (it != corpusVocabID2epochVocabID.end())
					//_nlzw[l][z][r] = (double) nlz[l][z] * phi_lzw[l][z][it->second];
					_nlzw[l][z][r] = nlzw[l][z][it->second];
				else {
					_nlzw[l][z][r] = 0;
				}
		    }
		}
	}

	// Update global count _ntlzw: first of all, need to update _ntlzw by poping out the oldest value _ntlzw[0] and poping in the current _nlzw value into _ntlzw[maxSlices-1], i.e., shift all the array elecments to to left
	for (t = maxSlices-1; t >= 0; t--)  { // t starting from '1' !!
		for (l = 0; l < numSentiLabs; l++) {
			for (z = 0; z < numTopics; z++) {
				for (r = 0; r < corpusVocabSize; r++) {
					if (t > 0)
						_ntlzw[t][l][z][r]= _ntlzw[t-1][l][z][r];
					else
						_ntlzw[0][l][z][r] = _nlzw[l][z][r];
				}
			}
		}
	}
	
	// compute _n_slzw[s][l][z][r]: total number of times word 'r' asssociated with sentiment label 'l' and topic label 'z' appear in the time scale from t\in[t-2^(s-1)+1, t]
	for (s = 1; s < numScales; s++) {
		for (l = 0; l < numSentiLabs; l++) {
			for (z = 0; z < numTopics; z++) {
				_n_slz[s][l][z] = 0.0;
				for (int r = 0; r < corpusVocabSize; r++) {   
					if (mode == MODE_SLIDING)
						_n_slzw[s][l][z][r] = _ntlzw[s-1][l][z][r];
					else if (mode == MODE_SKIP) {
						idx = pow((double)2, s-1)-1;
						_n_slzw[s][l][z][r] = _ntlzw[idx][l][z][r];
					}
					else if (mode == MODE_MULTISCALE) {
						tmp = 0.0;
						for (t = 0; t < pow((double)2, s-1); t++) 
							tmp += _ntlzw[t][l][z][r];
						_n_slzw[s][l][z][r] = tmp;
					}
					else
						_n_slzw[s][l][z][r] = 0;
					_n_slz[s][l][z] += _n_slzw[s][l][z][r];
				}
			}
		}
	}

	return 0;
}


void model::compute_pi_dl()
{	
	for (int m = 0; m < numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++) {
		    pi_dl[m][l] = (ndl[m][l] + gamma_dl[m][l]) / (nd[m] + gammaSum_d[m]);
		}
	}

}

void model::compute_theta_dlz()
{
	for (int m = 0; m < numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++)  {
			for (int z = 0; z < numTopics; z++) {
			    theta_dlz[m][l][z] = (ndlz[m][l][z] + alpha_lz[l][z]) / (ndl[m][l] + alphaSum_l[l]);    
			}
		}
	}
}


int model::save_model(string model_name)
{
	if (save_model_tassign(result_dir + model_name + tassign_suffix)) 
		return 1;
	
	if (save_model_twords(result_dir + model_name + twords_suffix)) 
		return 1;

	if (save_model_pi_dl(result_dir + model_name + pi_suffix)) 
		return 1;

	if (save_model_theta_dlz(result_dir + model_name + theta_suffix)) 
		return 1;

	/*if (save_model_phi_lzw(result_dir + model_name + phi_suffix)) 
		return 1; */

	if (save_model_others(result_dir + model_name + others_suffix)) 
		return 1;

	if (save_model_beta_lzw(result_dir + model_name + ".beta")) 
		return 1;

	/*if (opt_mu_mode == "EM") {
		if(save_model_mu(result_dir + model_name + ".mu")) 
	    return 1;
	}*/

	return 0;
}


int model::save_model_tassign(string filename)
{
    
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }

    // wirte docs with topic assignments for words
	for (int m = 0; m < pdataset->numDocs; m++) {    
		fprintf(fout, "%s \n", pdataset->pdocs[m]->docID.c_str());
		for (int n = 0; n < pdataset->pdocs[m]->length; n++) {
	        fprintf(fout, "%d:%d:%d ", pdataset->pdocs[m]->words[n], l[m][n], z[m][n]); //  wordID:sentiLab:topic
	    }
	    fprintf(fout, "\n");
    }

    fclose(fout);
    
	return 0;
}


int model::save_model_mu(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w"); 
    if (!fout) {
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
	}

	for (int l = 0; l < numSentiLabs; l++) {
		for (int z = 0; z < numTopics; z++) {
			fprintf(fout, "%d:%d \n", l, z);  // sentiLab:topic
			for (int s = 0; s < numScales; s++)
			    fprintf(fout, "%f ", mu_slz[s][l][z]);

			fprintf(fout, "\n");
		}
	}

	fclose(fout);

	return 0;
}


/*int model::save_model_mu(string filename)
{
	int currentStep = liter/updateParaStep - 1; // current saving step 
	bool fileExists;
	string muFile;
	muFile = filename;
	FILE * fout;

	// save the final mu file
	if (liter == niters) {
		string epochID =  pdataset->convertInt(pdataset->epochID);
		string filename = "epoch_" + epochID + "-final" + ".mu";

	  fout = fopen((result_dir + filename).c_str(), "w"); 
    if (!fout) {
			printf("Cannot open file %s to save!\n", (result_dir + filename).c_str());
			return 1;
		}
	}
	else if (currentStep == 0) {  // if this is the first time writing the file, remove the existing result files
		fout = fopen(muFile.c_str(), "w"); 
		if (!fout) { 
		    printf("Cannot open file %s to write!\n", muFile.c_str());
		    return 1;
		}
	}
	else {
		fout = fopen(muFile.c_str(), "a"); // use 'a' for append operation!!!!
    if (!fout) {
			printf("Cannot open file %s to save!\n", muFile.c_str());
			return 1;
		}

		fprintf(fout, "\n");
	}

	fprintf(fout, "%4d\n", liter);
	for (int l = 0; l < numSentiLabs; l++) {
    for (int z = 0; z < numTopics; z++) {
			for (int s = 0; s < numScales; s++)
		        fprintf(fout, "mu_lzs[%d][%d][%d]=%6.4g  ", l, z, s, mu_slz[s][l][z]);
			    
			fprintf(fout, "\n");
		}
	}

	fclose(fout);

	return 0;
}*/



// NOTE: there is only one epsilon file for each epoch
int model::save_model_epsilon()
{

	string epochID =  pdataset->convertInt(pdataset->epochID);
	string filename = "epoch_" + epochID + "-final" + ".epsilon";

	FILE * fout = fopen((result_dir + filename).c_str(), "w"); 

  if (!fout) {
		printf("Cannot open file %s to save!\n", (result_dir + filename).c_str());
		return 1;
	}

	for (int s = 0; s < numScales; s++) {
		for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++) {
				fprintf(fout, "%d:%d:%d \n", s, l, z);
				for (int r = 0; r < vocabSize; r++)
				    fprintf(fout, "%f ", epsilon_slzw[s][l][z][r]);
					    
				fprintf(fout, "\n");
			}
		}
	}

	fclose(fout);

	return 0;
}


int model::save_model_alpha(string filename)
{
  int currentStep = liter/updateParaStep - 1; // current saving step 
	string alphaFile;
	alphaFile = filename;
	FILE *fout;

	if (currentStep == 0)  // if this is the first time writing the file, remove the existing result files
	{
		fout = fopen(alphaFile.c_str(), "w"); 
		if (!fout) { 
		    printf("Cannot open file %s to write!\n", alphaFile.c_str());
		    return 1;
		}
	}
	
	else {
		fout = fopen(alphaFile.c_str(), "a"); // use 'a' for append operation!!!!
		if (!fout) {
			printf("Cannot open file %s to save!\n", alphaFile.c_str());
			return 1;
		}

		fprintf(fout, "\n");
	}
	
	fprintf(fout, "%4d\t", liter);
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) {
		    if (z == 0) fprintf(fout, "alpha[%d]={%8.4g ", l, alpha_lz[l][z]);
			else if (z == numTopics-1) fprintf(fout, "%8.4g", alpha_lz[l][z]);
			else fprintf(fout, "%8.4g ", alpha_lz[l][z]);
		}
		fprintf(fout, "}\t");
	}
	fclose(fout);

	return 0;
}


int model::save_model_time_duration(string filename)
{
	FILE *fout;
	
	if (epochID == 0) { 
		fout = fopen(filename.c_str(), "w"); 
		if (!fout) { 
			printf("Cannot open file %s to write!\n", filename.c_str());
			return 1;
		}
	}
	// append the processing time 
	else {
		fout = fopen(filename.c_str(), "a"); // use 'a' for append operation!!!!
        
		if (!fout) {
			printf("Cannot open file %s to update!\n", filename.c_str());
			return 1;
		}
	}
	
	fprintf(fout, "The processing time of epoch_%d is %g seconds. \n", epochID, time_duration);
	fclose(fout);

  return 0;
}


int model::save_model_twords(string filename) 
{   
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
    if (twords > vocabSize) 
	    twords = vocabSize; // equivalent to print out the whole vacabulary list
    
    mapid2word::iterator it; // typedef map<int, string> mapid2word, using map class
   
    for (int l = 0; l < numSentiLabs; l++) { 
        fprintf(fout, "Label %dth\n", l);
        for (int k = 0; k < numTopics; k++) { 
	        vector<pair<int, double> > words_probs;  // "words_probs"!!! what is the data structure of "words_probs"?
	        pair<int, double> word_prob;             // "word_prob"!!! different from above!!!
	        for (int w = 0; w < vocabSize; w++) { 
		        word_prob.first = w; // w is the word id 
	            word_prob.second = phi_lzw[l][k][w]; // the topic-word probability
	            words_probs.push_back(word_prob);
	        }
    
            // quick sort to sort word-topic probability
		    std::sort(words_probs.begin(), words_probs.end(), sort_pred());

	        fprintf(fout, "Topic %dth:\n", k);
	        for (int i = 0; i < twords; i++) { 
		        it = id2word.find(words_probs[i].first);
	            if (it != id2word.end()) 
			        fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
	        }
	    } // for topic
    } // for label
     
    fclose(fout);      
    return 0;    
}


int model::save_model_ntlzw(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
    }

	for (int t = 0; t < numScales; t++) {
		fprintf(fout, "t=%d (size: %d x %d x %d): \n", t, numSentiLabs, numTopics, vocabSize);
		for (int l = 0; l < numSentiLabs; l++) {
		    for (int z = 0; z < numTopics; z++) {
			    for (int r = 0; r < vocabSize; r++)
					fprintf(fout, "%f ", _ntlzw[t][l][z][r]);
			}
		}
		fprintf(fout, "\n");
	}

    fclose(fout);       

	return 0;
}


int model::save_model_pi_dl(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
    }

	for (int m = 0; m < numDocs; m++) { 
		fprintf(fout, "d_%d %s ", m, pdataset->pdocs[m]->docID.c_str());
		for (int l = 0; l < numSentiLabs; l++)
			fprintf(fout, "%f ", pi_dl[m][l]);
			
		fprintf(fout, "\n");
    }
   
    fclose(fout);       

	return 0;
}


int model::save_model_theta_dlz(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
    }
    
   for(int m = 0; m < numDocs; m++) {
       fprintf(fout, "Document %d\n", m);
	   for (int l = 0; l < numSentiLabs; l++) { 
	       for (int z = 0; z < numTopics; z++) 
		       fprintf(fout, "%f ", theta_dlz[m][l][z]);
		   
		   fprintf(fout, "\n");
		}
   }
   fclose(fout);
	
   return 0;
}


int model::save_model_phi_lzw(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
	for (int l = 0; l < numSentiLabs; l++) {  
	    for (int z = 0; z < numTopics; z++) { 
		    fprintf(fout, "Label:%d  Topic:%d\n", l, z);
     	    for (int r = 0; r < vocabSize; r++) 
			    fprintf(fout, "%.15f ", phi_lzw[l][z][r]);
	        
            fprintf(fout, "\n");
	   }
    }
    
    fclose(fout);    

	return 0;
}


int model::save_model_beta_lzw(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
	for (int l = 0; l < numSentiLabs; l++) {  
	    for (int z = 0; z < numTopics; z++) { 
		    fprintf(fout, "%d:%d\n", l, z);
     	    for (int r = 0; r < vocabSize; r++) 
			    fprintf(fout, "%f ", beta_lzw[l][z][r]);
	        
            fprintf(fout, "\n");
	   }
    }
    
    fclose(fout);    
	
	return 0;
}


int model::save_model_others(string filename) 
{
   
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
	fprintf(fout, "data_dir=%s\n", this->data_dir.c_str());
	fprintf(fout, "result_dir=%s\n", this->result_dir.c_str());
	fprintf(fout, "datasetFile=%s\n", this->datasetFile.c_str());
	fprintf(fout, "sentiLexFile=%s\n", this->sentiLexFile.c_str());

  fprintf(fout, "\n");
	fprintf(fout, "epochID=%d\n", epochID);
  fprintf(fout, "numDocs=%d\n", numDocs);
	fprintf(fout, "epochSize=%d\n", epochSize);
	fprintf(fout, "aveDocLength=%d\n", aveDocLength);
  fprintf(fout, "vocabSize=%d\n", vocabSize);
  fprintf(fout, "corpusVocabSize=%d\n", corpusVocabSize);
	fprintf(fout, "opt_mu_mode=%s\n", opt_mu_mode.c_str());
	fprintf(fout, "dydnamic mode=%d\n", mode);
	fprintf(fout, "numScales=%d\n", numScales);
	fprintf(fout, "maxSlices=%d\n", maxSlices);
	fprintf(fout, "numSentiLabs=%d\n", numSentiLabs);
	fprintf(fout, "numTopics=%d\n", numTopics);
	fprintf(fout, "liter=%d\n", liter);
	fprintf(fout, "savestep=%d\n", savestep);
	
	// alpha_old
	if (mode != MODE_NONE) {
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) {
		    if (z == 0) fprintf(fout, "alpha_old[%d]={", l);
		    fprintf(fout, "%.2g ", alpha_lz_old[l][z]);
		}
		fprintf(fout, "}\n");
	}
	}

	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) {
		    if (z == 0) fprintf(fout, "alpha[%d]={", l);
		    fprintf(fout, "%.2g ", alpha_lz[l][z]);
		}
		fprintf(fout, "}\n");
	}

	for (int l = 0; l < numSentiLabs; l++)
		fprintf(fout, "alphaSum[%d]=%g\n", l, alphaSum_l[l]);

	fprintf(fout, "_beta=%g\n", _beta);
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) // Only print out 3 topics
		    fprintf(fout, "betaSum[%d][%d]=%f\n", l, z, betaSum_lz[l][z]);
	}
	
	for (int l = 0; l < numSentiLabs; l++)
		fprintf(fout, "doc 0 gamma[%d]=%g\n", l, gamma_dl[0][l]);
	//fprintf(fout, "gammaSum=%g\n", gammaSum);
	
    fprintf(fout, "\n");
	
	//fprintf(fout, "------------- Lexicon statistics ------------\n");
	//fprintf(fout, "Lexicon file name: %s\n",sentiLexFile.c_str());
	//fprintf(fout, "Negation words file name: %s\n",negationFile.c_str());
	//fprintf(fout, "Total number negate2Positive=%d, negate2Negative=%d\n", posNegationCnt, negNegationCnt);
	//fprintf(fout, "Originally total number of lexicons in file %s :", sentiLexFile.c_str());
	//fprintf(fout, "pos = %d, neg = %d, neu = %d\n", numOriPosLex, numOriNegLex, numOriNeuLex);
	//fprintf(fout, "Total number of lexicons matches corpus without filtering:\n");
	//fprintf(fout, "pos = %d, neg = %d, neu = %d\n", numMatchedPos, numMatchedNeg, numMatchedNeu);
	//fprintf(fout, "Total number of lexicons incorporated after filtering with threshold:\n");
	//fprintf(fout, "pos = %d, neg = %d, neu = %d\n", numFltPos, numFltNeg, numFltNeu);
	//fprintf(fout, "Total number of times lexicons match the word tokens in corpus:\n");
	//fprintf(fout, "matchedPos = %d, matchedNeg = %d, matchedNeu = %d\n", matchedPosLexCnt, matchedNegLexCnt, matchedNeuLexCnt);

    fclose(fout);    
    
    return 0;
}


int model::init_estimate() 
{
    int sentiLab, topic;		
	
    // result_name = utils::generate_result_name(dfile, numTopics, _alpha, gamma, neuTH_L, neuTH_H, posTH_L, posTH_H, negTH_L, negTH_H); // generate the result name for current gamma and topic number settings
	// if (utils::make_dir(dir_results + result_name)) return 1; // Check whether the directory for saving results exists. Create directory if not.
	
	srand(time(0)); // initialize for random number generation
  	//seedMT( 1 + time(0) * 2 ); // seeding only works on uneven numbers
    
	z.resize(numDocs);
	l.resize(numDocs);
	
	for (int m = 0; m < numDocs; m++) {
		int docLength = pdataset->pdocs[m]->length;
		z[m].resize(docLength);
		l[m].resize(docLength);
          
	    // initialize z, l *****pdataset->pdocs[m]->words[n], for prior initialization, pos = label 0; neg = label 1, neutral = label 2
        for (int t = 0; t < docLength; t++) {
		    if (pdataset->pdocs[m]->words[t] < 0) {
			    printf("ERROE! word token %d has index smaller than 0 at doc[%d][%d]\n", pdataset->pdocs[m]->words[t], m, t);	
				return 1;
			}

		    // when numSentiLab < 3, incorporating priorSentilabs will cause error, 
		    // so need to check the condition of (pdataset->pdocs[m]->priorSentiLabels[t] < numSentiLabs)
    	    if ((pdataset->pdocs[m]->priorSentiLabels[t] > -1) && (pdataset->pdocs[m]->priorSentiLabels[t] < numSentiLabs)) {
			    sentiLab = pdataset->pdocs[m]->priorSentiLabels[t]; // incorporate prior information into the model
			    
			}
			else {
			    sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
	  		    //sentiLab = (int) ( (double) randomMT() * (double) numSentiLabs / (double) (4294967296.0 + 1.0) );
			    if (sentiLab == numSentiLabs) sentiLab = numSentiLabs -1;  // to avoid overflow, i.e., over the array boundary
			}
    	    l[m][t] = sentiLab; 
    	    
			// random initialize the topic assginment
			//topic = (int) ( (double) randomMT() * (double) numTopics / (double) (4294967296.0 + 1.0) );
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			if (topic == numTopics)  topic = numTopics - 1; // to avoid overflow, i.e., over the array boundary
			z[m][t] = topic; // z[m][t] here represents the n(th) word in the m(th) document
    	        	    
			// number of instances of word/term i assigned to sentiment label k and topic j
			nd[m]++;
			ndl[m][sentiLab]++;
			ndlz[m][sentiLab][topic]++;
			nlzw[sentiLab][topic][pdataset->pdocs[m]->words[t]]++;
			nlz[sentiLab][topic]++;
        } 
    }  // End for (int m = 0; m < numDocs; m++)
	
    return 0;
}


int model::estimate() 
{  
	int sentiLab, topic;
	mapname2labs::iterator it;
	bool fixPiFlag=false;
		
  //  if (twords > 0) {
		//// print out top words per topic
		//if (dataset::read_wordmap(dir_results+wordmapfile, &id2word)) {
		//	cout<< "Warning! Can not read wordmap " << dir_results+wordmapfile.c_str() << endl;
		//	return 1; 
		//}
  //  }

	// beta_slz = \sum_s[mu_slz * epsilon_slzw] -- Note: must update \beta for the first use before updating \mu_slz
	/*if (epochID != 0)
	 {
	     update_beta(); 
	}*/

	printf("Processing epoch_%d!\n", epochID);
	printf("Sampling %d iterations!\n", niters);

	for (liter = 1; liter <= niters; liter++) {
	  printf("Iteration %d ...\n", liter);
	
		for (int m = 0; m < numDocs; m++) {
			// check whether use fixed pi (known doc labs)
 			it = docLabs.find(pdataset->pdocs[m]->docID);
                	if (it != docLabs.end()) fixPiFlag=true;
			else fixPiFlag = false;

		    for (int n = 0; n < pdataset->pdocs[m]->length; n++) {
					sampling(m, n, sentiLab, topic, fixPiFlag);
					l[m][n] = sentiLab; 
					z[m][n] = topic; 
				} 
		}
		
		if (updateParaStep > 0 && liter % updateParaStep == 0) { 
			string epochID = pdataset->convertInt(this->epochID);
			update_Parameters();
			//if(save_model_alpha(result_dir + "epoch_" + epochID + ".alpha")) return 1;
			//if(save_model_mu(result_dir + "epoch_" + epochID + ".mu")) return 1;
		}
		
		if (savestep > 0 && liter % savestep == 0) {
			if (liter == niters) break;
	    // saving the model
	    printf("Saving the model at iteration %d ...\n", liter);
	    compute_pi_dl();
	    compute_theta_dlz();
	    compute_phi_lzw();
	    save_model(putils->generate_model_name(liter, epochID));
		}
	}  // End: for (liter = last_iter + 1; liter <= niters + last_iter; liter++) 
	
	printf("Gibbs sampling completed!\n");
	printf("Saving the final model!\n");
	
	compute_pi_dl();
	compute_theta_dlz();
	compute_phi_lzw();

	save_model(putils->generate_model_name(-1, epochID));
	
	// update \alpha_t-1 and E_t-1 for next epoch
	//compute_phi_lzw();
	if (mode != MODE_NONE) {
		compute_expected_counts();
		compute_epsilon_slzw();
		//save_model_epsilon();
		update_alpha_lz_old();
	}
	//save_model_ntlzw(result_dir + utils::generate_model_name(liter, epochID) + ".ntlzw");

	return 0;
}


int model::sampling(int m, int n, int& sentiLab, int& topic, bool fixPiFlag)
{
	sentiLab = l[m][n];
	topic = z[m][n];
	int w = pdataset->pdocs[m]->words[n]; // the ID/index of the current word token in vocabulary 
  double u;
	
	nd[m]--;
	ndl[m][sentiLab]--;
	ndlz[m][sentiLab][topic]--;
	nlzw[sentiLab][topic][pdataset->pdocs[m]->words[n]]--;
	nlz[sentiLab][topic]--;


	// do multinomial sampling via cumulative method, and p_st[l][k] is the temp variable for sampling
	for (int l = 0; l < numSentiLabs; l++) {
		for (int k = 0; k < numTopics; k++) {
			if (!fixPiFlag) {
				p[l][k] = (nlzw[l][k][w] + beta_lzw[l][k][w]) / (nlz[l][k] + betaSum_lz[l][k]) *
		   		(ndlz[m][l][k] + alpha_lz[l][k]) / (ndl[m][l] + alphaSum_l[l]) *
				(ndl[m][l] + gamma_dl[m][l]) / (nd[m] + gammaSum_d[m]);
			}
			else {
				p[l][k] = (nlzw[l][k][w] + beta_lzw[l][k][w]) / (nlz[l][k] + betaSum_lz[l][k]) *
		   		(ndlz[m][l][k] + alpha_lz[l][k]) / (ndl[m][l] + alphaSum_l[l]) *
				(gamma_dl[m][l]) / (gammaSum_d[m]);
			}
		}
	}
	
	// accumulate multinomial parameters
	for (int l = 0; l < numSentiLabs; l++)  {    //sentiment label l must be > 1;  
		for (int k = 0; k < numTopics; k++) {
			if (k==0)  {   // the first element of an sub array
			    if (l==0) continue;
		      else p[l][k] += p[l-1][numTopics-1]; // accumulate the sum of the previous array
			}
			else p[l][k] += p[l][k-1];
		}
	}
	// scaled sample because of unnormalized p_st[] ***--The normalization is been done here!!!--***
	u = ((double)rand() / RAND_MAX) * p[numSentiLabs-1][numTopics-1];

	bool loopBreak=false;
	// here we get the sample of label, from [0, S-1]
	for (sentiLab = 0; sentiLab < numSentiLabs; sentiLab++) {   
		for (topic = 0; topic < numTopics; topic++) { 
		  if (p[sentiLab][topic] > u) {
		  	loopBreak = true;
		  	break;  
		  }
		}
		if (loopBreak == true) 
			break;
	}
    
	if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1; // the max value of label is (S - 1)!!!
	if (topic == numTopics) topic = numTopics - 1; 

	// add newly estimated z_i to count variables
	nd[m]++;
	ndl[m][sentiLab]++;
	ndlz[m][sentiLab][topic]++;
	nlzw[sentiLab][topic][pdataset->pdocs[m]->words[n]]++;
	nlz[sentiLab][topic]++;

  return 0;  
}


void model::update_alpha_lz_old()
{
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++)
			alpha_lz_old[l][z] = alpha_lz[l][z];
	}
}


int model::update_Parameters()
{
	if (mode != MODE_NONE && useHistoricalAlpha && epochID > 0 && liter/updateParaStep == 1) {
		// optimize alpha
		for (int l = 0; l < numSentiLabs; l++)
			optimize_alpha_lz(l, MAX_ITERATION);
	}
	else {
		int ** data; // temp valuable for exporting 3-dimentional array to 2-dimentional 
		double * alpha_temp;
		data = new int*[numTopics];
		for (int k = 0; k < numTopics; k++) {
			data[k] = new int[numDocs];
			for (int m = 0; m < numDocs; m++) {
				data[k][m] = 0;
			}
		}

		alpha_temp = new double[numTopics];
		for (int k = 0; k < numTopics; k++) 
			alpha_temp[k] = 0.0;

		// update \alpha 
		for (int j = 0; j < numSentiLabs; j++) {		
			// import data from ntldsum to data
			for (int k = 0; k < numTopics; k++) {
				for (int m = 0; m < numDocs; m++) {	
					data[k][m] = ndlz[m][j][k]; // ntldsum[j][k][m];
				}
			}

			// import alpha
			for (int k = 0; k < numTopics; k++)
				alpha_temp[k] =  alpha_lz[j][k];//alpha[j][k];

			polya_fit_simple(data, alpha_temp, numTopics, numDocs);

			// update alpha
			alphaSum_l[j] = 0.0;
			for (int k = 0; k < numTopics; k++) {
				alpha_lz[j][k] = alpha_temp[k];
				alphaSum_l[j] += alpha_lz[j][k];
			}	
		}
	}

	//optimize mu;
	if (mode != MODE_NONE && opt_mu_mode == "EM" && epochID > 0) {
		for (int l = 0; l < numSentiLabs; l++) {
			for (int z = 0; z < numTopics; z++)
				optimize_mu_slz(l, z, MAX_ITERATION);
		}
		
		if (update_beta())  // beta is correlated with mu, so we need to update beta after optimizing mu
			return 1;
	}
	
	return 0;
}


int model::optimize_mu_slz(int sentiLab, int topic, int maxIteration)
{
	int ifault1, ifault2;
	double Asum; // A_sum_lz = \sum_d[digamma(ndlz + alpha_lz) - digamma(alpha_lz)]
	double Bsum; // B_sum_lz = \sum_d[digamma(ndl + \sum_z(alpha_lz)) - digamma(\sum_z(alpha_lz))]
	bool converge_flag = false; 
	double C_s, mu_sum; 
	map<int, int>::iterator it;
		
	for (int i = 0; i < maxIteration; i++) {
		if (i != 0) {
			// update mu_slz with the optimized value after each iteration	
			mu_lz[sentiLab][topic] = 0.0;
			for (int s = 0; s < numScales; s++) {
				mu_slz[s][sentiLab][topic] = opt_mu_slz[s][sentiLab][topic];
				mu_lz[sentiLab][topic] += mu_slz[s][sentiLab][topic];
			}
		}

		Bsum = digama(nlz[sentiLab][topic] + mu_lz[sentiLab][topic], &ifault1) - digama(mu_lz[sentiLab][topic], &ifault2);

		for (int s = 0; s < numScales; s++) {
			double numerator = 0.0;
			C_s = 0.0;
			for (int r = 0; r < vocabSize; r++) {
				it = epochVocabID2corpusVocabID.find(r);
				if (it == epochVocabID2corpusVocabID.end())	{
					printf("Error: optimize_mu_slz: word [%d] not found in corpus vocab list\n", r);
					return 1;
				}
					
				mu_sum = 0.0;
				for (int t = 0; t < numScales; t++) 
					mu_sum += 	mu_slz[t][sentiLab][topic] * epsilon_slzw[t][sentiLab][topic][it->second];
				
    		Asum = digama(nlzw[sentiLab][topic][r] + mu_sum, &ifault1) - digama(mu_sum, &ifault2);
				C_s += epsilon_slzw[s][sentiLab][topic][it->second] * Asum;
			}
			
			numerator = mu_slz[s][sentiLab][topic] * C_s;

			opt_mu_slz[s][sentiLab][topic] = numerator / Bsum;
		}

		// normailze the weight of mu
		double weightSum = 0.0;
		for (int s = 0; s < numScales; s++)
		    weightSum += opt_mu_slz[s][sentiLab][topic];

		for (int s = 0; s < numScales; s++)
		    opt_mu_slz[s][sentiLab][topic] = opt_mu_slz[s][sentiLab][topic] / weightSum;

		// terminate iteration ONLY if each dimension of {mu_1, mu_2, ... mu_S} satisfy the termination criteria,  
		for (int _s = 0; _s < numScales; _s++) {
		    if (fabs(mu_slz[_s][sentiLab][topic] - opt_mu_slz[_s][sentiLab][topic]) > 0.001)  break;
				if (_s == numScales-1)  converge_flag = true; 
		}
	
		if (converge_flag) {
		    printf("Optimizing mu converged at iteration %d\n", i);
	      break;
		}
		else if (i == maxIteration-1) cout<<"mu_slz haven't converged after maximum iteration: "<<i+1<<endl;
	}

	return 0;
}


int model::optimize_alpha_lz(int sentiLab, int maxIteration)
{
	//int topic;
	int ifault1, ifault2;
	double Asum; // A_sum_lz = \sum_d[digamma(ndlz + alpha_lz) - digamma(alpha_lz)]
	double Bsum; // B_sum_lz = \sum_d[digamma(ndl + \sum_z(alpha_lz)) - digamma(\sum_z(alpha_lz))]
	bool converge_flag = false; 

	for (int i = 0; i < maxIteration; i++) {
		if (i != 0) {
		// update alpha_lz with the optimized value after each iteration	
			alphaSum_l[sentiLab] = 0.0;
			for (int k = 0; k < numTopics; k++) {
				alpha_lz[sentiLab][k] = opt_alpha_lz[sentiLab][k];
				alphaSum_l[sentiLab] += alpha_lz[sentiLab][k];
			}   
		}

		for (int z = 0; z < numTopics; z++) {
			// update Asum
		    Asum = 0.0;
		    for (int m = 0; m < numDocs; m++)
		        Asum += (digama(ndlz[m][sentiLab][z] + alpha_lz[sentiLab][z], &ifault1) - digama(alpha_lz[sentiLab][z], &ifault2));
		    
			// update BSsum
			Bsum = 0.0;
			for (int m = 0; m < numDocs; m++)
			    Bsum += (digama(ndl[m][sentiLab] + alphaSum_l[sentiLab], &ifault1) - digama(alphaSum_l[sentiLab], &ifault2));
				
			// calculate optimal alpha value
			opt_alpha_lz[sentiLab][z] = (upsilon*alpha_lz_old[sentiLab][z] - 1 + alpha_lz[sentiLab][z]*Asum) / (upsilon + Bsum);
		}

		// terminate iteration ONLY if each dimension of {alpha_1, alpha_2, ... alpha_k} satisfy the termination criteria,  
		for (int k = 0; k < numTopics; k++) {
		    if (fabs(alpha_lz[sentiLab][k] - opt_alpha_lz[sentiLab][k]) > 0.00001)  
		    	break;  // >0.000001
			
			if (k == numTopics-1)  
				converge_flag = true; 
		}
	
		if (converge_flag) {
		    printf("Optimizing alpha converded at iteration %d\n", i);
	        break;
		}
		else if (i == maxIteration-1) cout<<"Optimizing alpha haven't converged after maximum iteration: "<<i+1<<endl;
	}
	
	return 0; 
}


int model::set_mu_decayFunction()
{
	vector<double> weight(numScales); 
	double lambda = 0.5;
	double weightSum = 0.0;
	int s;

	// we only calculate the weight for s\in[1, numScales], i.e., we do not consider the case of s == 0
	for (s = 0; s < numScales; s++)  {
		weight[s] = exp(-lambda * s);
		weightSum += weight[s];
	}

	for (s = 0; s < numScales; s++)
		mu_s[s] = weight[s]/weightSum; 

	return 0;
}

int model::read_doc_labels(string docLabFile) 
{
	docLabs.clear();
	char buff[BUFF_SIZE_SHORT];
    string line;
    vector<double> labels;

    
    FILE * fin = fopen(docLabFile.c_str(), "r");
    if (!fin) {
		printf("Cannot open file %s to read!\n", docLabFile.c_str());
		return 1;
    }    
     
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin) != NULL) {
		line = buff;
		
		strtokenizer strtok(line, " \t\r\n");
		
		if (strtok.count_tokens() != numSentiLabs+1)  { // no. of doc labels should = numSentiLabs  
			printf("Error! The number of doc labels in the file %s line %s does not match the set number of sentiment labels %d!\n", docLabFile.c_str(), buff, numSentiLabs);
			return 1;
		}
	
		labels.clear();	
		for (int i = 1; i < strtok.count_tokens(); i++) 
			labels.push_back(atof(strtok.token(i).c_str()));

		docLabs.insert(pair<string, vector<double> >(strtok.token(0), labels));
    }
    
	if (docLabs.size() <= 0) {
		printf("Can not find any document labels in file %s!\n", docLabFile.c_str());
		return 1;
	}

    fclose(fin);
    return 0;
}
