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
	   Revised by Yulan He, 03/06/2011
*/
   
   
#ifndef	_MODEL_H
#define	_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include "dataset.h"
#include "document.h"
#include "map_type.h"
#include "utils.h"
#include "math_func.h"
#include "polya_fit_simple.h"
#include "strtokenizer.h"


using namespace std;



class model
{
public:

	model(void);
	~model(void);
	
	// parameter that will contribute to the new epoch or will be the same for all the epochs 
	mapword2atr word2atr;
	mapid2word id2word; 
	mapword2prior sentiLex; // <string, Word_Prior > => <word, [senti lab, word prior distribution]>
	mapname2labs docLabs; // <string, vector<double> > => <doc ID, doc label distribution>
	
	vector<string> newWords;
	string data_dir;
	string result_dir;
	string datasetFile;
	string sentiLexFile;
	string docLabFile;
	string wordmapfile;
	string tassign_suffix;
	string pi_suffix;
	string theta_suffix;
	string phi_suffix;
	string others_suffix;
	string twords_suffix;
	string opt_mu_mode;
	int mode;

	int numTopics;
	int numSentiLabs; 
	int numScales; 
	int maxSlices;
	int niters;
	int liter;
	int twords;
	int savestep;
	int updateParaStep;
	int epochID;
	int epochLength; 
	double _alpha;
	double _beta;
	double _gamma;
	double time_duration;

	bool useHistoricalAlpha;

	// corpus parameter
	int corpusVocabSize;
	mapword2id corpusVocab;   // <string, int> ==> <word, corpusWideVocabID>
	map<int, int> epochVocabID2corpusVocabID;
	map<int, int> corpusVocabID2epochVocabID;
	
	// init parameter functions
	// int set_default_values(int model_status, int numDocs, int numTopics, int numSentiLabs, int vocabSize, int corpusSize, int aveDocLength, int numScales, vector<string> docs, document ** pdocs);
	int init(int argc, char ** argv);
	
	int excute_model();
	void update_alpha_lz_old();
	

private:
	//*********** epoch specific parameters that shoud be deleted after the end of the epoch training ************  
	
	int numDocs;
	int vocabSize;
	int epochSize;
	int aveDocLength;
	
	ifstream fin;	
	dataset * pdataset;
	utils * putils;

	// model real counts
	vector<int> nd;
	vector<vector<int> > ndl;
	vector<vector<vector<int> > > ndlz;
	vector<vector<vector<int> > > nlzw;
	vector<vector<int> > nlz;
	
	// posterial and topic label assignments
	vector<vector<double> > p;
	vector<vector<int> > z;
	vector<vector<int> > l;
	
	// expected counts 
	vector<vector<vector<double> > > _nlzw;   // size: (L x T x V) -- the expected number of times word 'w' associated with sentiment label l and topic z at epoch t.
	vector<vector<vector<vector<double> > > > _n_slzw;  // size: (numScales x L x T x V)
	vector<vector<vector<double> > > _n_slz;   //size: (numScales x L x T)
	
	// model parameters
	vector<vector<double> > pi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > theta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > phi_lzw; // size: (L x T x V) -- this is also the expected_\phi
	
	// hyperparameters 
	vector<vector<double> > alpha_lz; // \alpha_tlz size: (L x T) -- the hyperparameter for \theta
	vector<double> alphaSum_l; 
	vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
	vector<vector<double> > betaSum_lz;
	vector<vector<double> > gamma_dl; // size: (numDocs x L)
	vector<double> gammaSum_d; 
	double upsilon; // parameter of the gamma function -- set to '1'
	vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information 
		
	// parameters for updating \alpha and \mu 
		
	vector<vector<double> > opt_alpha_lz;  //optimal value, size:(L x T) -- this is the optimal value of alpha_lz after fix point iteration
	vector<vector<vector<double> > > opt_mu_slz;   //optimal value  size: (numScales x L x T) -- this is the optimal value of mu_slz after fix point iteration 
		
	//*********** END: epoch specific parameters that shoud be deleted after the end of the epoch training ************  
	
	
	// evolutionary parameters
	vector<double> mu_s;    // for decay parameter over s time slices
	vector<vector<double> >mu_lz;
	vector<vector<vector<double> > > mu_slz; // size: (numScales x L x T)
	vector<vector<vector<double> > > epsilon_lzw;  // size: L x T x V  
	vector<vector<vector<vector<double> > > > epsilon_slzw; // size: (numScales x L x T x V);
	
	vector<vector<double> >  alpha_lz_old;
	vector<vector<vector<vector<double> > > > _ntlzw; // size: (2^(numScales-1) x L x T x V) -- the expected number of times word 'w' associated with sentiment label l and topic z at epoch t.
	
	
	/************************* Functions ***************************/
	int read_doc_labels(string docLabFile) ;
	int set_gamma();
	
	int set_ntlzw_epsilon();
	int init_dynamic_mode_parameters();
	int init_model_parameters();
	int reset_model_parameters();
	int reset_dynamic_mode_parameters();
	int set_dynamic_hyperparameters();
	inline int delete_model_parameters() {
		numDocs = 0;
		vocabSize = 0;
		epochSize = 0;
		aveDocLength = 0;
		
		if (pdataset != NULL) {
			delete pdataset;
			pdataset = NULL;
		}
		
		return 0;
	}


	// estimate functions
	
	int init_estimate();
	int estimate();
	int prior2beta(); // for incorporating priro information for the first time
	int newWords_prior2beta();
	int sampling(int m, int n, int& sentiLab, int& topic, bool fixPiFlag);
	
	// compute parameter functions
	int compute_expected_counts();
	int compute_epsilon_slzw();
	void compute_pi_dl(); 
	void compute_theta_dlz(); 
	void compute_phi_lzw(); 
	
	// update parameter functions
	void init_parameters();
	
	int update_Parameters();
	//int update_mu_slz();
	//int update_mu_lz();
	//int update_mu_lz(int scale, int sentiLab, int topic);
	int update_beta(); // update beta based on mu_slz and epsilon_slzw
	//int update_beta(int scale, int sentiLab, int topic);
	
	int optimize_alpha_lz(int sentiLab, int maxIteration);
	int optimize_mu_slz(int sentiLab, int topic, int maxIteration);
	int set_mu_decayFunction();
	double compute_C_s(int scale, int sentiLab, int topic);
	double compute_B_lz(int sentiLab, int topic);
	

	// save model parameter funtions 
	int save_model(string model_name);
	int save_model_tassign(string filename);
	int save_model_pi_dl(string filename);
	int save_model_theta_dlz(string filename);
	int save_model_phi_lzw(string filename);
	int save_model_others(string filename);
	int save_model_twords(string filename);
	int save_model_hyperPara(string filename);
	int save_model_beta_lzw(string filename);
	int save_model_ntlzw(string filename);
	int save_model_time_duration(string filename);
	int save_model_alpha(string filename);
	int save_model_mu(string filename);
	int save_model_epsilon();

	//int update_expcted_ntlzw(); // i.e., _ntlzw. Update after processing each epoch. 
};

#endif
