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
*/
   
   
#ifndef _INFERENCE_H
#define _INFERENCE_H

#include <sys/stat.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "constants.h"
#include "document.h"
#include "dataset.h"
#include "utils.h"
#include "strtokenizer.h"

using namespace std; 


class Inference
{
public:
	Inference(void);
	~Inference(void);

	//***********************************************
	//**** parameters loaded from previous model ****
	//***********************************************
	int numScales;
	int maxSlices;
    int numSentiLabs; 
	int numTopics;
	int numDocs; 
	int vocabSize; 

	// model real counts -- loaded from previous model
	// vector<int> nd;
	// vector<vector<int> > ndl;
	// vector<vector<vector<int> > > ndlz;
	vector<vector<vector<int> > > nlzw;
	vector<vector<int> > nlz;

	//***********************************************
	//**************** END of Loading ***************
	//***********************************************



	// ********************* For New Document Inference *******************

	// parameter that will contribute to the new epoch or will be the same for all the epochs 
    mapword2atr word2atr;
	mapword2id word2id; 
	mapid2word id2word; 
	//mapword2id word2id;
    map<int, int> id2_id;
	map<int, int> _id2id;

	mapword2prior sentiLex; // <string, int> => <word, polarity>

	
	vector<string> newWords;

	string model_dir;
	string data_dir;
	string result_dir;
	string datasetFile;
	string sentiLexFile;
	string wordmapfile;
	// string muFile;
	// string epsilonFile;
	string betaFile;

	string tassign_suffix;
    string pi_suffix;
    string theta_suffix;
    string phi_suffix;
    string others_suffix;
    string twords_suffix;
	string model_name;

	dataset * pmodelData;	// pointer to training dataset object
    dataset * pnewData; // pointer to new dataset object
	utils * putils;
	
	//document ** docs;
	//document ** newdocs;

	int newNumDocs;
	int newVocabSize; 

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
	
	vector<vector<double> > new_p; // for posterior
	vector<vector<int> > new_z;
    vector<vector<int> > new_l;
	vector<vector<int> > z;  // for loaded model
    vector<vector<int> > l;  // for loaded model 


	// model real counts -- from NEW documents
	vector<int> new_nd;
	vector<vector<int> > new_ndl;
	vector<vector<vector<int> > > new_ndlz;
	vector<vector<vector<int> > > new_nlzw;
	vector<vector<int> > new_nlz;


	// hyperparameters 
    vector<vector<double> > alpha_lz; // \alpha_tlz size: (L x T) -- the hyperparameter for \theta
	vector<double> alphaSum_l; 
	vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
	vector<vector<double> > betaSum_lz;
	vector<double> gamma_l; // size: (L)
	double gammaSum; 
	vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information 
	

	// model parameters
	vector<vector<double> > newpi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > newtheta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > newphi_lzw; // size: (L x T x V) -- this is also the expected_\phi


	// evolutionary parameters 
	//vector<vector<double> >mu_lz;
    //vector<vector<vector<double> > > mu_slz; // size: (numScales x L x T)
    //vector<vector<vector<vector<double> > > > epsilon_slzw; // size: (numScales x L x T x V);



	// functions 
	int init(int argc, char ** argv);

	
	// init for inference
    int init_inf();
    // inference for new (unseen) data based on the estimated LDA model
    int inference();
    int inf_sampling(int m, int n, int& sentiLab, int& topic);
	int init_parameters();
    
	int read_newData(string filename);
	int load_model(string model_name);
	int read_betaFile(string filename);
	int read_para_setting(string filename);
	int read_alpha(string filename);
	int prior2beta(); // for incorporating priro information
		
	// int read_muFile(string muFile);
	// int read_epsilonFile(string filename);

	
	// compute model parameters
	void compute_newpi();
	void compute_newtheta();
	int compute_newphi();
	double compute_perplexity();


	// save new data models
	int save_model(string model_name);
    int save_model_newtassign(string filename);
    int save_model_newpi_dl(string filename);
    int save_model_newtheta_dlz(string filename);
    int save_model_newphi_lzw(string filename);
    int save_model_newothers(string filename);
    int save_model_newtwords(string filename); 
	int save_model_perplexity(string filename); 
};

#endif
