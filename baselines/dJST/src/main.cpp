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
   

#include "model.h"
#include "inference.h"
#include "utils.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <iostream>
#include <map>


using namespace std; 

void show_help();


int main(int argc, char ** argv) 
{
	int model_status = MODEL_STATUS_UNKNOWN;
	
	utils *putils = new utils();

	model_status = putils->parse_args(argc, argv, model_status);
	
	if (putils)
		delete putils;

	if (model_status == MODEL_STATUS_UNKNOWN) {
		printf("Please specify the task you would like to perform, training (-est) or inference (-inf)!\n");
		show_help();
		return 1;
	}
	else if (model_status == MODEL_STATUS_EST){
		model DJST;
	
		//The model class is initialised here
		if (DJST.init(argc, argv)) {
			show_help();
			return 1;
		}
		
		if(DJST.excute_model()) return 1;
	}
	else if (model_status == MODEL_STATUS_INF) {
		Inference DJST;
		
		if (DJST.init(argc, argv)) {
			show_help();
			return 1;
		}
	}

	
	return 0;
}


void show_help() 
{
	printf("Command line usage:\n");
	printf("djst -est|-inf [options]\n");
	printf("-est \t Estimate the DJST model from scratch.\n");
	printf("-inf \t Perform inference on unseen (new) data using a trained model.\n");
	
	printf("\n-----------------------------------------------------------\n");
	printf("Command line opitions:\n\n");
	printf("-config \t Configuration file.\n");
	printf("-mode \t\t Estimate the DJST model with sliding, skip or multiScale mode.\n");
	printf("-nscales \t The length of the past epoch history that DJST accounts for. The default is 4.\n");
	printf("-nsentiLabs \t The number of sentiment labels. The default is 3.\n");
	printf("-ntopics \t The number of topics. The default is 50.\n");
	printf("-niters \t The number of Gibbs sampling iterations. The default is 1000.\n");
	printf("-savestep \t The step (counted by the number of Gibbs sampling iterations) at which the model is saved to hard disk. The default is 200.\n");
	printf("-updateParaStep The step (counted by the number of Gibbs sampling iterations) at which the hyperparameters are updated. The default is 40.\n");
	printf("-twords \t The number of most likely words to be printed for each topic. The default is 20.\n");
	printf("-opt_mu_mode \t The way to set the evolutionary matrix weights, either decay or EM. The default is decay.\n");
	printf("-data_dir \t The directory where the input training data is stored.\n");
	printf("-result_dir \t The directory where the output models and parameters will be stored.\n");
	printf("-datasetFile \t The input training data file.\n");
	printf("-sentiFile \t The sentiment lexicon file.\n");
	printf("-vocab \t\t The vocabulary file.\n");
	printf("-alpha \t\t The hyperparameter of the per-document sentiment specific topic proportion. The default is avgDocLength*0.05/(numSentiLabs*numTopics).\n");
	printf("-beta \t\t The hyperparameter of the per-corpus sentiment specific topic-word distribution. The default is 0.01.\n");
	printf("-gamma \t\t The hyperparameter of the per-document sentiment proportion. The default is avgDocLength*0.05/numSentiLabs.\n");
	printf("-useHistoricalAlpha \t Whether alpha of the current epoch should be initialised as the one in the previous epoch. The default is FALSE.\n");
	printf("-docLabFile \t The file containing training document labels. Document labels are used to set gamma.\n");
	printf("-model \t\t The name of the previously trained model. (for inference only).\n");
}

