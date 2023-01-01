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
	 	Revised by Yulan He, 06/05/2011
*/

#include <stdio.h>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include "strtokenizer.h"
#include "utils.h"
#include "model.h"
#include "inference.h"
#include "dataset.h"
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

using namespace std;

// --- define WINDOWS
#undef WINDOWS
#ifdef _WIN32
  #define WINDOWS
#endif
#ifdef __WIN32__
  #define WINDOWS
#endif

#ifdef WINDOWS
	#include <direct.h>  // For _mkdir().
	#include <io.h>   // For access().
#else 
	#include <unistd.h>   // For access().
#endif

utils::utils()
{
	model_status = MODEL_STATUS_UNKNOWN;
	mode_str = "";
	model_dir = "";
	data_dir = "";
	result_dir = "";
  model_name = "";
	configfile = "";
	wordmapfile = "";
	sentiLexFile = "";
	docLabFile = "";
  datasetFile = "";
  configfile = "";
	opt_mu_mode = "decay";
	numScales = 0; 
  numSentiLabs = 0;
	numTopics = 0;
  niters = 0;
  savestep = 0;
  twords = 0;
	updateParaStep = -1; 
	epochLength = -1;
	alpha = -1.0;
	beta = -1.0;
  gamma = -1.0;    // this is for the convience of parsing the command line arguements.
	useHistoricalAlpha = false;	
}

int utils::parse_args(int argc, char ** argv, int&  model_status)
{
	int i = 1; 
  while (i < argc) {
		string arg = argv[i];
			
		if (arg == "-est") {
			model_status = MODEL_STATUS_EST;
			break;
		}
		else if (arg == "-estc") {
			model_status = MODEL_STATUS_ESTC;
			break;
		}
		else if (arg == "-inf") {
			model_status = MODEL_STATUS_INF;
			break;
		}

		i++;
	}

	return (model_status);
}


int utils::parse_args(int argc, char ** argv, model * pmodel) 
{
	// read configuration file first if exist
	int i = 1; 
  while (i < argc) {
		string arg = argv[i];
			
		if (arg == "-config") {
			configfile = argv[++i];
			break;
		}
		i++;
	}
	if (configfile != "") {
		if (read_config_file(configfile)) return 1;
	}
	
    i = 1; 
    while (i < argc) {
		string arg = argv[i];
			
		if (arg == "-est")
			model_status = MODEL_STATUS_EST;
		else if (arg == "-estc")
			model_status = MODEL_STATUS_ESTC;
		else if (arg == "-inf")
			model_status = MODEL_STATUS_INF;

		else if (arg == "-mode") 
			mode_str = argv[++i];	    
			
		else if (arg == "-model_dir") 
			data_dir = argv[++i];	    

		else if (arg == "-data_dir") 
			data_dir = argv[++i];	    
				
		else if (arg == "-result_dir") 
			result_dir = argv[++i];	    

		else if (arg == "-datasetFile") 
			datasetFile = argv[++i];	    
				
		else if (arg == "-sentiFile") 
			sentiLexFile = argv[++i];
		
		else if (arg == "-docLabFile") 
			docLabFile = argv[++i];

		else if (arg == "-opt_mu_mode") 
			opt_mu_mode = argv[++i];
						
		else if (arg == "-model") 
			model_name = argv[++i];	    	    
				
		else if (arg == "-nscales")
			numScales = atoi(argv[++i]);	   

		else if (arg == "-nsentiLabs")
			numSentiLabs = atoi(argv[++i]);	   

		else if (arg == "-ntopics")
			numTopics = atoi(argv[++i]);	
				
		else if (arg == "-epochLength")
			epochLength = atoi(argv[++i]);

		else if (arg == "-niters")
			niters = atoi(argv[++i]);	    
				
		else if (arg == "-savestep") 
			savestep = atoi(argv[++i]);           

		else if (arg == "-twords")
			twords = atoi(argv[++i]);
			
		else if (arg == "-vocab") 
			wordmapfile = argv[++i];

		else if (arg == "-updateParaStep") 
			updateParaStep = atoi(argv[++i]);

		else if (arg == "-alpha") 
			alpha = atof(argv[++i]);
				
		else if (arg == "-beta") 
			beta = atof(argv[++i]);	

		else if (arg == "-gamma") 
			gamma = atof(argv[++i]);

		else if (arg == "-useHistoricalAlpha")
			useHistoricalAlpha = true;
			
		else if (arg == "-config")
			configfile = argv[++i];

		else {
			printf("Unknown command line option [%s]\n", argv[i]);
			return 1;
		}
		i++;				    
	}	
	
	if (model_status == MODEL_STATUS_UNKNOWN) {
		printf("Please specify the task you would like to perform, model training (-est) or inference (-inf)!\n");
		return 1;
	}
	
	// dynamic modeling choices
	if (mode_str == "sliding")
		pmodel->mode = MODE_SLIDING;
	else if (mode_str == "skip")
		pmodel->mode = MODE_SKIP;
	else if (mode_str == "multi-scale")
		pmodel->mode = MODE_MULTISCALE;
	else if (mode_str != "") {
		printf("Please specify the dynamic modeling mode, eith \'sliding\' or \'skip\' or \'multi-scale\'!\n");
		return 1;
	}
	else
		pmodel->mode = MODE_NONE;
		
	if (wordmapfile != "")   
		pmodel->wordmapfile = wordmapfile;
			
	if (sentiLexFile != "")
		pmodel->sentiLexFile = sentiLexFile;
	
	if (docLabFile != "")
		pmodel->docLabFile = docLabFile;

	if (datasetFile != "") {
		pmodel->datasetFile = datasetFile;
	}

	if (opt_mu_mode != "EM" && opt_mu_mode != "decay") {
		printf("Please specify the evolutionary matrix weight estimation mode, eith \'EM\' or \'decay\'!\n");
		return 1;
	}
	else
		pmodel->opt_mu_mode = opt_mu_mode;

	if (data_dir != "")	{
		if (data_dir[data_dir.size() - 1] != '/') data_dir += "/";
		pmodel->data_dir = data_dir;
	}
	else {
		printf("Please specify input data dir!\n");
		return 1;
	}
	
	if (result_dir != "")	{
		if (make_dir(result_dir)) return 1;
	  if (result_dir[result_dir.size() - 1] != '/') result_dir += "/";
		pmodel->result_dir = result_dir;
	}
	else {
		printf("Please specify output dir!\n");
		return 1;
	}

	/*if (negationFile != "")
		pmodel->negationFile = negationFile;
		
	  if (datasetType != "")
		pmodel->datasetType = datasetType;

	if (dir_data == "")	pmodel->dir_data = "./data/";
	*/
			
    if (model_status == MODEL_STATUS_EST) {
		//pmodel->dir = dir;
			
		//string::size_type idx = dfile.find_last_of("/");			
		//if (idx == string::npos)
		//    pmodel->dir = "./";
		//else {
		 //   pmodel->dir = dfile.substr(0, idx + 1);
		  //  pmodel->dfile = dfile.substr(idx + 1, dfile.size() - pmodel->dir.size());
		   // printf("dir = %s\n", pmodel->dir.c_str());
		   // printf("dfile = %s\n", pmodel->dfile.c_str());
	//	}
		
		//pmodel->model_status = model_status;
					
    	if (numScales > 0) pmodel->numScales = numScales+1;
		if (numSentiLabs > 0) pmodel->numSentiLabs = numSentiLabs;
		if (numTopics > 0) pmodel->numTopics = numTopics;
		if (epochLength > 0) pmodel->epochLength = epochLength;
		if (updateParaStep > 0) pmodel->updateParaStep = updateParaStep;
		if (niters > 0)  pmodel->niters = niters;
		if (savestep > 0) pmodel->savestep = savestep;
		if (twords > 0)   pmodel->twords = twords;
		
		
		if (alpha > 0.0) pmodel->_alpha = alpha; 
		if (beta > 0.0) pmodel->_beta = beta;
		if (gamma > 0.0) pmodel->_gamma = gamma;
							
		pmodel->useHistoricalAlpha = useHistoricalAlpha;
			
		/*
		if (boostCnt >= 0) pmodel->boostCnt = boostCnt;

		pmodel->boostFlag = boostFlag;
		pmodel->estHyperFlag = estHyperFlag;
		pmodel->updateGammaFlag = updateGammaFlag;
		if (neuTH_L > 0) pmodel->neuTH_L = neuTH_L;
		if (neuTH_H > 0) pmodel->neuTH_H = neuTH_H;
		if (posTH_L > 0) pmodel->posTH_L = posTH_L;
		if (posTH_H > 0) pmodel->posTH_H = posTH_H;
		if (negTH_L > 0) pmodel->negTH_L = negTH_L;
		if (negTH_H > 0) pmodel->negTH_H = negTH_H;
		*/
    } 
    
    else if (model_status == MODEL_STATUS_ESTC) 
    {
  		
    }  
    
    else if (model_status == MODEL_STATUS_INF) 
    {
  		
    }

    return 0;
}
   

int utils::parse_args(int argc, char ** argv, Inference * pmodel_inf) 
{
	// read configuration file first if exist
	int i = 1; 
	while (i < argc) {
		string arg = argv[i];
			
		if (arg == "-config") {
			configfile = argv[++i];
			break;
		}
		i++;
	}
	if (configfile != "") {
		if (read_config_file(configfile)) return 1;
	}
	
    i = 1; 
    while (i < argc) {
		string arg = argv[i];
			
		if (arg == "-est")
			model_status = MODEL_STATUS_EST;
		else if (arg == "-estc")
			model_status = MODEL_STATUS_ESTC;
		else if (arg == "-inf")
			model_status = MODEL_STATUS_INF;
		
		else if (arg == "-model_dir") 
			model_dir = argv[++i];	    
		
		else if (arg == "-data_dir") 
			data_dir = argv[++i];	    
				
		else if (arg == "-result_dir") 
			result_dir = argv[++i];	    
		
		else if (arg == "-datasetFile") 
			datasetFile = argv[++i];	    
				
		else if (arg == "-sentiFile") 
			sentiLexFile = argv[++i];
						
		else if (arg == "-model") 
			model_name = argv[++i];	    	    
				
		else if (arg == "-niters")
			niters = atoi(argv[++i]);	    
				
		else if (arg == "-savestep") 
			savestep = atoi(argv[++i]);           
		
		else if (arg == "-twords")
			twords = atoi(argv[++i]);
			
		else if (arg == "-vocab") 
			wordmapfile = argv[++i];
		
		else if (arg == "-updateParaStep") 
			updateParaStep = atoi(argv[++i]);
		
		else if (arg == "-alpha") 
			alpha = atof(argv[++i]);
				
		else if (arg == "-beta") 
			beta = atof(argv[++i]);	

		else if (arg == "-gamma") 
			gamma = atof(argv[++i]);
			
              else if (arg == "-config")
                        configfile = argv[++i];

		else {
			printf("Unknown command line option [%s]\n", argv[i]);
			return 1;
		}
				
		i++;	
				    
	}	

    if (model_status == MODEL_STATUS_UNKNOWN) {
		printf("Please specify the task you would like to perform, model training (-est) or inference (-inf)!\n");
		return 1;
    }
    
	if (wordmapfile != "") 
		pmodel_inf->wordmapfile = wordmapfile;
		
	if (sentiLexFile != "") 
		pmodel_inf->sentiLexFile = sentiLexFile;
	//else {
	//	printf("Please specify sentiment lexicon file!\n");
	//	return 1;
	//}
	
	if (datasetFile != "")
		pmodel_inf->datasetFile = datasetFile;
	else {
		printf("Please specify input dataset file!\n");
		return 1;
	}
	
	if (model_dir != "")	{
		if (model_dir[model_dir.size() - 1] != '/') model_dir += "/";
		pmodel_inf->model_dir = model_dir;
	}
	
	if (data_dir != "")	{
		if (data_dir[data_dir.size() - 1] != '/') data_dir += "/";
		pmodel_inf->data_dir = data_dir;
	}
	else {
		printf("Please specify input data dir!\n");
		return 1;
	}
	
	if (result_dir != "")	{
		if (make_dir(result_dir)) return 1;
		if (result_dir[result_dir.size() - 1] != '/') result_dir += "/";
		pmodel_inf->result_dir = result_dir;
	}
	else {
		printf("Please specify output dir!\n");
		return 1;
	}
	
	if (model_name != "")
		pmodel_inf->model_name = model_name;
	else {
		printf("Please specify the trained dJST model name!\n");
		return 1;
	}
    
	if (niters > 0) 
		pmodel_inf->niters = niters;
	else 
		// default number of Gibbs sampling iterations for doing inference
		pmodel_inf->niters = 20;
	
	if (twords > 0) pmodel_inf->twords = twords;
	if (savestep > 0) pmodel_inf->savestep = savestep;
	if (updateParaStep > 0) pmodel_inf->updateParaStep = updateParaStep;
	if (alpha > 0.0) pmodel_inf->_alpha = alpha; 
	if (beta > 0.0) pmodel_inf->_beta = beta;
	if (gamma > 0.0) pmodel_inf->_gamma = gamma;
	
	// read <model>.others file to assign values for ntopics, alpha, beta, etc.
	if (read_and_parse(pmodel_inf->model_dir + pmodel_inf->model_name + ".others", pmodel_inf)) 
		return 1;
    
    return 0;
}
   
int utils::read_config_file(string filename) 
{
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
		printf("Cannot open file: %s\n", filename.c_str());
		return 1;
    }
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
			line = buff;
			strtokenizer strtok(line, "= \t\r\n");
			int count = strtok.count_tokens();
	
			// invalid, ignore this line
			if (count != 2)
				continue;
			
			string optstr = strtok.token(0);
			string optval = strtok.token(1);
			
			if (optstr == "mode")
				mode_str = optval;				
			else if (optstr == "nscales") 
				numScales = atoi(optval.c_str());
			else if(optstr == "nsentiLabs") 
				numSentiLabs = atoi(optval.c_str());
			else if(optstr == "ntopics") 
				numTopics = atoi(optval.c_str());	
			else if(optstr == "niters") 
				niters = atoi(optval.c_str());	
			else if(optstr == "savestep") 
				savestep = atoi(optval.c_str());				
			else if (optstr == "updateParaStep") 
				updateParaStep = atoi(optval.c_str());
			else if(optstr == "twords") 
				twords = atoi(optval.c_str());	
			else if(optstr == "data_dir") 
				data_dir = optval;	
			else if (optstr == "model_dir") 
				model_dir = optval;	    
			else if(optstr == "result_dir") 
				result_dir = optval;	
			else if(optstr == "datasetFile") 
				datasetFile = optval;	
			else if(optstr == "sentiFile") 
				sentiLexFile = optval;	
			else if(optstr == "docLabFile") 
				docLabFile = optval;					
			else if (optstr == "vocabFile") 
				wordmapfile = optval;					
			else if(optstr == "opt_mu_mode") 
				opt_mu_mode = optval;																																				
			else if (optstr == "alpha")
				alpha = atof(optval.c_str());
			else if (optstr == "beta")    
				beta = atof(optval.c_str());
			else if (optstr == "gamma")    
				gamma = atof(optval.c_str());
			else if (optstr == "useHistoricalAlpha")    
				useHistoricalAlpha = true;
			else if (optstr == "model")  
				model_name = optval;
		} // END while
		
		fclose(fin);
		    
    return 0;
}


int utils::read_and_parse(string filename, Inference * pmodel_inf) 
{
    // open file <model>.others to read:
    // alpha=?
    // beta=?
    // nsubj=?
    // nsenti=?
    // ndocs=?
    // nwords=?
    // citer=? // current iteration (when the model was saved)
    
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
		printf("Cannot open file: %s\n", filename.c_str());
		return 1;
    }
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
		line = buff;
		strtokenizer strtok(line, "= \t\r\n");
		int count = strtok.count_tokens();

		// invalid, ignore this line
		if (count != 2)
			continue;
		
		string optstr = strtok.token(0);
		string optval = strtok.token(1);
		
		/*if (optstr == "alpha") {
			pmodel->alpha = atof(optval.c_str());
		}
	
		else if (optstr == "beta") {	    
			pmodel->beta = atof(optval.c_str());
		}*/
		
		if (optstr == "numScales") 
			pmodel_inf->numScales = atoi(optval.c_str());
		else if (optstr == "epochID") 
				pmodel_inf->epochID = atoi(optval.c_str());
		else if (optstr == "maxSlices") 
				pmodel_inf->maxSlices = atoi(optval.c_str());
		else if(optstr == "numSentiLabs") 
				pmodel_inf->numSentiLabs = atoi(optval.c_str());
		else if(optstr == "numTopics") 
				pmodel_inf->numTopics = atoi(optval.c_str());			
		else if (optstr == "numDocs")  
				pmodel_inf->numDocs = atoi(optval.c_str());
		else if (optstr == "vocabSize") 
				pmodel_inf->vocabSize = atoi(optval.c_str());
		else if (optstr == "updateParaStep") 
				pmodel_inf->updateParaStep = atoi(optval.c_str());
		else if (optstr == "liter") 
				pmodel_inf->liter = atoi(optval.c_str());
	
	} // END while
		
	fclose(fin);
		    
    return 0;
}


string utils::generate_model_name(int iter) 
{
	string model_name;
	std::stringstream out;
	
	char buff[BUFF_SIZE_SHORT];
	
	sprintf(buff, "%05d", iter);
	
	if (iter >= 0)
		model_name = buff;
	else
		model_name = "final";
	
	return model_name;
}

string utils::generate_model_name(int iter, int epochID) 
{
	string epoch_name;
	string model_name;
	std::stringstream out;
	
	out << epochID;
	epoch_name = out.str();
	epoch_name = "epoch_" + epoch_name + "-";
	
	char buff[BUFF_SIZE_SHORT];
	
	sprintf(buff, "%05d", iter);
	
	if (iter >= 0) 
		model_name = epoch_name + buff;
	else
		model_name = epoch_name + "final";
    
    return model_name;
}


string utils::generate_result_name(string dfile, int numTopics, double alpha, double gamma[], int neuTH_L, int neuTH_H, int posTH_L, int posTH_H, int negTH_L, int negTH_H) 
{	
	string result_name;
	string dat_name; 
	string temp;
	size_t pos; 
	
	pos = dfile.find("."); // i.e. for 'MR.dat', get rid of '.dat' and retain 'MR' only. 
	if (pos > 0)
		dat_name = dfile.substr(0, pos);  
	else
		dat_name = dfile;
				
	char buff[BUFF_SIZE_SHORT];
	
	sprintf(buff, "t%d_g%.2g_%.2g_%.2g_a%.2g", numTopics, gamma[0], gamma[1], gamma[2], alpha); // all the element of alpha is identical, so just use the the value of alphap[0] to represent vector alpha			
	result_name = buff;
	
	if (neuTH_L>0 || neuTH_H>0) {
		sprintf(buff, "neuTH_%d_%d", neuTH_L, neuTH_H);
		temp = buff;
	}
	
	if (posTH_L>0 || posTH_H>0) {
		sprintf(buff, "posTH_%d_%d", posTH_L, posTH_H);
		temp += buff;
	}
	
	if (negTH_L>0 || negTH_H>0) {
		sprintf(buff, "negTH_%d_%d", negTH_L, negTH_H);
		temp += buff;
	}
	
	result_name += temp;
	
	// append 'MR' in the begining of 'result_name'
	result_name = dat_name + "_" + result_name; 

    return result_name;
}

#ifdef WINDOWS
int utils::make_dir(string strPath) 
{
	if(_access(strPath.c_str(), 0) == 0) 
		return 0;
	else if(_mkdir(strPath.c_str()) == 0) 
		return 0;
	else {
		printf("Problem creating dirctory %s !\n",strPath.c_str());
		return 1;
	} 
}
#else
int utils::make_dir(string strPath) 
{ 
	if(access(strPath.c_str(), 0) == 0)  
		return 0;
	else if(mkdir(strPath.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0)
		return 0;
	else { 
		cout<<"Problem creating dirctory "<<strPath.c_str()<<endl;
		return 1; 
	}
}
#endif


//void utils::quicksort(vector<pair<int, double> > & vect, int left, int right) {
//    int l_hold, r_hold;
//    pair<int, double> pivot;
//    
//    l_hold = left;
//    r_hold = right;    
//    int pivotidx = left;
//    pivot = vect[pivotidx];
//
//    while (left < right) {
//	while (vect[right].second <= pivot.second && left < right) {
//	    right--;
//	}
//	if (left != right) {
//	    vect[left] = vect[right];
//	    left++;
//	}
//	while (vect[left].second >= pivot.second && left < right) {
//	    left++;
//	}
//	if (left != right) {
//	    vect[right] = vect[left];
//	    right--;
//	}
//    }
//
//    vect[left] = pivot;
//    pivotidx = left;
//    left = l_hold;
//    right = r_hold;
//    
//    if (left < pivotidx) {
//	quicksort(vect, left, pivotidx - 1);
//    }
//    if (right > pivotidx) {
//	quicksort(vect, pivotidx + 1, right);
//    }    
//}

