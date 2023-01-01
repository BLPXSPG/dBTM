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
           Revised by Yulan He, 13/05/2011
*/
   
   
#include "inference.h"



using namespace std;

Inference::Inference(void)
{
	//**** parameters should be loaded from previous model -- all set to '0' for init ****//
	numScales = 0;
    numSentiLabs = 0; 
	numTopics = 0;
	numDocs = 0; 
	vocabSize = 0; 
	_beta = -1.0;
	
	//*** parameters that don't have to be loaded from previous model *** //
	wordmapfile = "wordmap.txt";
    tassign_suffix = ".newtassign";
    pi_suffix = ".newpi";
    theta_suffix = ".newtheta";
    phi_suffix = ".newphi";
    others_suffix = ".newothers";
    twords_suffix = ".newtwords";
	model_name = "";
	data_dir = "";
	result_dir = "";
	datasetFile = "";
	sentiLexFile = "";
	//muFile = "";
	//epsilonFile = "";
	betaFile = "";

	updateParaStep = -1;
	savestep = 20;
	twords = 20;
	niters = 20;
	
	putils = new utils();

	pmodelData = NULL;
	pnewData = NULL;
}


Inference::~Inference(void)
{
	if (putils)
		delete putils;
		
	if (pmodelData)
		delete pmodelData;
	
	if (pnewData)
		delete pnewData;
}


int Inference::init(int argc, char ** argv) 
{
  // call parse_args
  if (putils->parse_args(argc, argv, this))
		return 1;
    
	cout<<"data_dir = "<<data_dir<<endl;
	cout<<"result_dir = "<<result_dir<<endl;
	cout<<"datasetFile = "<<datasetFile<<endl;
	cout<<"sentiLexFile = "<<sentiLexFile<<endl;
	cout<<"model_dir = "<<model_dir<<endl;
	cout<<"model_name = "<<model_name<<endl;
	cout<<"wordmapfile = "<<wordmapfile<<endl;
	//cout<<"numScales = "<<numScales<<endl;
	//cout<<"maxSlices = "<<maxSlices<<endl;
	cout<<"numTopics = "<<numTopics<<endl;
	cout<<"numSentiLabs = "<<numSentiLabs<<endl;
	cout<<"niters = "<<niters<<endl;
	cout<<"savestep = "<<savestep<<endl;
	cout<<"twords = "<<twords<<endl;
	//cout<<"updateParaStep = "<<updateParaStep<<endl;
	//cout<<"epochLength = "<<epochLength<<endl;

	if(init_inf()) {
	    printf("Throw expectation in init_inf()!  \n");
		return 1; 
	}

	if(inference()) {
	    printf("Throw expectation in inference()!  \n");
		return 1; 
	}

    return 0;
}


//int Inference::read_epsilonFile(string filename)
//{
//
//	int numLines = (numScales+1) * numSentiLabs * numTopics; 
//
//	epsilon_slzw.resize(numScales+1);
//	for (int s = 0; s < (numScales+1); s++)
//	{
//		epsilon_slzw[s].resize(numSentiLabs);
//		for (int l = 0; l < numSentiLabs; l++)
//		{
//			epsilon_slzw[s][l].resize(numTopics);
//			for (int z = 0; z < numTopics; z++)
//			{
//				epsilon_slzw[s][l][z].resize(vocabSize);
//			}
//		}
//	}
//
//
//	FILE * fin = fopen(filename.c_str(), "r");
//    if (!fin) 
//	{
//	    printf("Cannot open file %s to read!\n", filename.c_str());
//	    return 1;
//    }    
//    
//	for (int i = 0; i < numLines; i++)
//	{
//
//		// read the dimension labels 
//		char buff[BUFF_SIZE_LONG];
//		string line;
//    
//		fgets(buff, BUFF_SIZE_LONG - 1, fin);
//		line = buff; 
//	
//		strtokenizer strtok(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
//		if (strtok.count_tokens() != 3) 
//		{
//			printf("Warning! Line %d in %s contains less than 4 words!\n", filename.c_str());
//			return 1; 
//		}
//
//		int scale = atoi(strtok.token(0).c_str());
//		int sentiLab = atoi(strtok.token(1).c_str());
//		int topic = atoi(strtok.token(2).c_str());
//   
//   
//		// read the data value 
//		fgets(buff, BUFF_SIZE_LONG - 1, fin);
//		line = buff; 
//	
//		strtokenizer values(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
//		if (values.count_tokens() != vocabSize) 
//		{
//			printf("Warning! Line %d in %s contains words less than vocabSize!\n", filename.c_str());
//			return 1; 
//		}
//
//		for (int r = 0; r < vocabSize; r++)
//		{
//			epsilon_slzw[scale][sentiLab][topic][r] = atof(values.token(r).c_str());
//		}
//
//
//	} // END  for numLines; 
//
//
//	//for (int s = 0; s < (numScales+1); s++)
//	//{
//	//	for (int l = 0; l < numSentiLabs; l++)
//	//	{
//	//		for (int t = 0; t < numTopics; t++)
//	//		{
//	//		    for (int r = 0; r < vocabSize; r++)
//	//			{
//	//				printf("epsilon_slzw[%d][%d][%d][%d] = %g\n", s, l, t, r, epsilon_slzw[s][l][t][r]);
//	//			}
//	//		}
//	//	}
//	//}
//
//
//
//	return 0;
//}


//int Inference::read_muFile(string muFile)
//{
//	char buff[BUFF_SIZE_LONG];
//	string line;
//	int numLines = numSentiLabs * numTopics; 
//
//
//	mu_slz.resize(numScales+1);
//	for (int s = 0; s < (numScales+1); s++)
//	{
//		mu_slz[s].resize(numSentiLabs);
//		for (int l = 0; l < numSentiLabs; l++)
//		{
//			mu_slz[s][l].resize(numTopics);
//		}
//	}
//
//
//	FILE * fin = fopen(muFile.c_str(), "r");
//    if (!fin) 
//	{
//	    printf("Cannot open file %s to read!\n", muFile.c_str());
//	    return 1;
//    }    
//    
//	for (int i = 0; i < numLines; i++)
//	{
//
//		// read the dimension labels 
//		fgets(buff, BUFF_SIZE_LONG - 1, fin);
//		line = buff; 
//	
//		strtokenizer strtok(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
//		if (strtok.count_tokens() != 2) 
//		{
//			printf("Warning! Line %d in %s contains less than 2 words!\n", muFile.c_str());
//			return 1; 
//		}
//
//		int sentiLab = atoi(strtok.token(0).c_str());
//		int topic = atoi(strtok.token(1).c_str());
//		
//   
//   
//		// read the data value 
//		fgets(buff, BUFF_SIZE_LONG - 1, fin);
//		line = buff; 
//	
//		strtokenizer values(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
//		if (values.count_tokens() != (numScales+1)) 
//		{
//			printf("Warning! Line %d in %s contains words less than vocabSize!\n", muFile.c_str());
//			return 1; 
//		}
//
//		for (int s = 0; s < (numScales+1); s++)
//		{
//			mu_slz[s][sentiLab][topic] = atof(values.token(s).c_str());
//		}
//
//
//	} // END  for numLines; 
//
//
//	//for (int z = 0; z < numTopics; z++)
//	//{
//	//    for (int l = 0; l < numSentiLabs; l++)
//	//	{
//	//	    for (int s = 0; s < (numScales+1); s++)
//	//		{
//	//		    printf("mu_slz[%d][%d][%d] = %g\n", s, l, z, mu_slz[s][l][z]);
//	//		}
//	//	}
//	//}
//	
//
//
//	return 0;
//}

int Inference::read_para_setting(string filename)
{
	char buff[BUFF_SIZE_LONG];
	string line;
	//ifstream fin;
	
	numSentiLabs = 0;
	numTopics = 0;
	
	FILE * fin = fopen(filename.c_str(), "r");
  if (!fin)  
  {
    printf("Cannot open file %s to read!\n", filename.c_str());
    return 1;
  }    
    
	while ( fgets(buff, BUFF_SIZE_LONG - 1, fin) != NULL)
	{
		//fin >> line;
		
		line = buff; 
		
		strtokenizer values(line, ": \t\r\n={}[]"); // \t\r\n is the data data separator, so called white space

		if (values.token(0) == "numSentiLabs")
	    numSentiLabs = atoi(values.token(1).c_str());
	  else if (values.token(0) == "numTopics")
	    numTopics = atoi(values.token(1).c_str());
	    
	  if (numSentiLabs > 0 && numTopics > 0)
	  	break; 	
	}

	fclose(fin);
	
	if (numSentiLabs == 0 || numTopics == 0) {
		cout << "Cannot find the settings of numSentiLabs or numTopics in " << filename << endl;
		return 1;
	}
		
	return 0;
}

int Inference::read_alpha(string filename)
{
	string line;
	ifstream fin;
	
	fin.open(filename.c_str());
	if (fin.fail())
	{ 
		cout << "Cannot open file " << filename << " to read!\n";
		return 1;
	}

	int i = 0;
	while ( !fin.eof()  )
	{
		fin >> line;
		
		strtokenizer values(line, ": \t\r\n={}[]"); // \t\r\n is the data data separator, so called white space

		if (values.token(0) == "alpha") {
			if (atoi(values.token(1).c_str()) == i) {
				for (int j = 0; j < numTopics; j++) {
					alpha_lz[i][j] = atof(values.token(2+j).c_str());
				}
				i++;
			}
		}
		
		if (i >= numSentiLabs) break;
	}

	fin.close();
	
	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
		for (int z = 0; z < numTopics; z++)
			alphaSum_l[l] += alpha_lz[l][z];
	}
		
	return 0;
}

int Inference::read_betaFile(string filename)
{
	char buff[BUFF_SIZE_LONG];
	string line;
	int numLines = numSentiLabs * numTopics; 


	FILE * fin = fopen(filename.c_str(), "r");
    if (!fin)  
    {
	    printf("Cannot open file %s to read!\n", filename.c_str());
	    return 1;
    }    
    
	for (int i = 0; i < numLines; i++)
	{
		// read the dimension labels 
		fgets(buff, BUFF_SIZE_LONG - 1, fin);
		line = buff; 
	
		strtokenizer strtok(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
		if (strtok.count_tokens() != 2) 
		{
			printf("Warning! Line %d in %s contains %d words!\n", i, betaFile.c_str(), strtok.count_tokens());
			return 1; 
		}

		int sentiLab = atoi(strtok.token(0).c_str());
		int topic = atoi(strtok.token(1).c_str());
		 
		// read the data value 
		fgets(buff, BUFF_SIZE_LONG - 1, fin);
		line = buff; 
	
		strtokenizer values(line, ": \t\r\n"); // \t\r\n is the data data separator, so called white space
		if (values.count_tokens() != vocabSize) 
		{
			printf("Warning! Line %d in %s contains %d words. Not equal to vocabSize=%d!\n", i, betaFile.c_str(), values.count_tokens(), vocabSize);
			return 1; 
		}

		for (int r = 0; r < vocabSize; r++)
			beta_lzw[sentiLab][topic][r] = atof(values.token(r).c_str());
	} // END  for numLines; 

	
	for (int l = 0; l < numSentiLabs; l++)    
	{                                         
		for (int z = 0; z < numTopics; z++)   
		{
			betaSum_lz[l][z] = 0.0;
			for (int r = 0; r < vocabSize; r++)
			    betaSum_lz[l][z] += beta_lzw[l][z][r];
		}
	}

	return 0;
}


// read the tassign file 
int Inference::load_model(string filename) 
{
    char buff[BUFF_SIZE_LONG];
	string line;
    
    //string filename = result_dir + model_name + tassign_suffix;
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) 
	{
	    printf("Cannot open file %s to load model!\n", filename.c_str());
	    return 1;
    }
    
	// allocate memory for z and pmodelData
	pmodelData->pdocs = new document*[numDocs];	
	pmodelData->numDocs = numDocs; 
	pmodelData->vocabSize= vocabSize;
	l.resize(pmodelData->numDocs);
	z.resize(pmodelData->numDocs);


	//int numLines = numDocs * 2; 
    for (int m = 0; m < numDocs; m++) 
	{
		fgets(buff, BUFF_SIZE_LONG - 1, fin);  // first time just ignore the document ID
		fgets(buff, BUFF_SIZE_LONG - 1, fin);  // second time read the sentiment label / topic assignments
		line = buff; 

	    strtokenizer strtok(line, " \t\r\n");
	    int length = strtok.count_tokens();
	
	    vector<int> words;
		vector<int> sentiLabs;
	    vector<int> topics;

	    for (int j = 0; j < length; j++) 
		{
	        string token = strtok.token(j);
    
	        strtokenizer tok(token, ":");
	        if (tok.count_tokens() != 3) 
			{
		        printf("Invalid word-sentiment-topic assignment line!\n");
		        return 1;
	        }
	    
	        words.push_back(atoi(tok.token(0).c_str()));
			sentiLabs.push_back(atoi(tok.token(1).c_str()));
	        topics.push_back(atoi(tok.token(2).c_str()));
	    }
	
		// allocate and add new document to the corpus
		document * pdoc = new document(words);
		pmodelData->add_doc(pdoc, m);

		//printf("length of Doc[%d] = %d \n", m, pmodelData->pdocs[m]->length);
		//printf("vocabSize = %d \n", pmodelData->vocabSize);

		// assign values for l
		l[m].resize(sentiLabs.size());
		for (int j = 0; j < (int)sentiLabs.size(); j++) 
			l[m][j] = sentiLabs[j];
		
		// assign values for z
		z[m].resize(topics.size());
		for (int j = 0; j < (int)topics.size(); j++) 
			z[m][j] = topics[j];
	}  // END for (m)

    fclose(fin);
    
	// init the counts for loaded model
	nlzw.resize(numSentiLabs); // need to distingusi 
	for (int l = 0; l < numSentiLabs; l++)
	{
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
		{
			nlzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++)
			    nlzw[l][z][r] = 0;
		}
	}

	nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
	{
		nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
            nlz[l][z] = 0;
	}

	// assign values for nlzw and nlz 
	for (int m = 0; m < pmodelData->numDocs; m++)
	{
		int docLength = pmodelData->pdocs[m]->length;
		
		for (int n = 0; n < docLength; n++)
		{
			int w = pmodelData->pdocs[m]->words[n];
			int sentiLab = this->l[m][n];
			int topic = this->z[m][n];

			nlzw[sentiLab][topic][w]++;
			nlz[sentiLab][topic]++;
		}
	}
	
    return 0;
}

int Inference::prior2beta()
{
	mapword2atr::iterator wordIt;
	mapword2prior::iterator sentiIt;

	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		wordIt = word2atr.find(sentiIt->first);
		//cout << sentiIt->first << endl;
		if (wordIt != word2atr.end()) {
			//cout << "word matched = " << sentiIt->first << endl;
			for (int j = 0; j < numSentiLabs; j++)  {
				lambda_lw[j][wordIt->second.id] = sentiIt->second.labDist[j];
				//cout << sentiIt->second.labDist[j] << " ";
			}
			//cout << endl;
		}
	}

	// Note: the 'r' index of lambda[j][r] is corresponding to the vocabulary ID. 
	// Therefore the correct prior info can be incorporated to corresponding word cound nlzw,
	// as 'w' is also corresponding to the vocabulary ID. 
  for (int l = 0; l < numSentiLabs; l++) {                                          
		for (int z = 0; z < numTopics; z++) {
			betaSum_lz[l][z] = 0.0;
		    for (int r = 0; r < pnewData->vocabSize; r++) {
			    beta_lzw[l][z][r] = beta_lzw[l][z][r] * lambda_lw[l][r];  
			    betaSum_lz[l][z] += beta_lzw[l][z][r];
		    }
		}
	}

	return 0;
}

int Inference::init_inf()
{
	pmodelData = new dataset; 
	pmodelData->epochID = this->epochID;
	string epochID = pmodelData->convertInt(pmodelData->epochID);
	
	// read in number of topics and number of sentiment labels
	if(read_para_setting(model_dir + model_name + ".others")) 
	{
	  printf("Error to read para setting of numSentiLabs and numTopics!\n");
		return 1; 
	}
	
	// load model here
	if(load_model(model_dir + model_name + ".tassign")) 
	{
	    printf("Error to load_model!\n");
		return 1; 
	}

	if(read_newData(data_dir + datasetFile))  //(data_dir+datasetFile).c_str());
	{
	    printf("Error to read test data!\n");
		return 1; 
	}

	if(init_parameters())
	{
	    printf("Error to init_parameters!\n");
		return 1; 
	}

	/*if(read_alpha(model_dir + model_name + ".others"))
	{
	  printf("Error to read alpha settings!\n");
		return 1; 
	}*/
	
	/*if(read_betaFile(model_dir + model_name + ".beta"))
	{
	    printf("Error to read betaFile!\n");
		return 1; 
	}*/

	printf("New documents statistics: \n");
	printf("numDocs = %d\n", pnewData->numDocs);
	printf("vocabSize = %d\n", pnewData->vocabSize);
	printf("numNew_word = %d\n", (int)(pnewData->newWords.size()));

	// init inf
	int sentiLab, topic; 
	new_z.resize(pnewData->numDocs);
	new_l.resize(pnewData->numDocs);

	for (int m = 0; m < pnewData->numDocs; m++)
	{
		int docLength = pnewData->_pdocs[m]->length;
		new_z[m].resize(docLength);
		new_l[m].resize(docLength);
		
		for (int t = 0; t < docLength; t++) 
		{
		    if (pnewData->_pdocs[m]->words[t] < 0) 
			{
			    printf("ERROE! word token %d has index smaller than 0 at doc[%d][%d]\n", pnewData->_pdocs[m]->words[t], m, t);	
				return 1;
			}

			// sample sentiment label
   		if ((pnewData->pdocs[m]->priorSentiLabels[t] > -1) && (pnewData->pdocs[m]->priorSentiLabels[t] < numSentiLabs)) {
			    sentiLab = pnewData->pdocs[m]->priorSentiLabels[t]; // incorporate prior information into the model  
			}
			else {
			    sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
			    if (sentiLab == numSentiLabs) sentiLab = numSentiLabs -1;  // to avoid overflow, i.e., over the array boundary
			}
    
	  	new_l[m][t] = sentiLab;

			// sample topic label 
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			if (topic == numTopics)  topic = numTopics - 1; // to avoid overflow, i.e., over the array boundary
			new_z[m][t] = topic; // z[m][t] here represents the n(th) word in the m(th) document

			// number of instances of word/term i assigned to sentiment label k and topic j
			new_nd[m]++;
			new_ndl[m][sentiLab]++;
			new_ndlz[m][sentiLab][topic]++;
			new_nlzw[sentiLab][topic][pnewData->_pdocs[m]->words[t]]++;
			new_nlz[sentiLab][topic]++;
       } 
	}  // End for (int m = 0; m < numDocs; m++)
	
	return 0;
}


int Inference::inference()
{
	int sentiLab, topic;
		
	printf("Processing epoch_%d!\n", epochID);
	printf("Sampling %d iterations for inference!\n", niters);

	liter = 0; 
	for (liter = 1; liter <= niters; liter++) {
		printf("Iteration %d ...\n", liter);
	
		for (int m = 0; m < pnewData->numDocs; m++) {
			for (int n = 0; n < pnewData->pdocs[m]->length; n++) {
				inf_sampling(m, n, sentiLab, topic);
				new_l[m][n] = sentiLab; 
				new_z[m][n] = topic; 
			} 
		}
		
		if (savestep > 0 && liter % savestep == 0) {
			if (liter == niters) break;
				
			// saving the model
			printf("Saving the model at iteration %d ...\n", liter);
			compute_newpi();
			compute_newtheta();
			compute_newphi();
			compute_perplexity();
			
			save_model(model_name + "_" + putils->generate_model_name(liter));
		} // End: if (savestep > 0) 
	}  // End: for (liter = 1; liter <= niters; liter++) 
    
	printf("Gibbs sampling completed!\n");
	printf("Saving the final model!\n");

	compute_newpi();
	compute_newtheta();
	compute_newphi();

	save_model(model_name + "_" + putils->generate_model_name(-1));

	return 0;
}


int Inference::init_parameters()
{
	// init parameters
	new_p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
	{
		new_p[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
		    new_p[l][z] = 0.0;
	}

	new_nd.resize(pnewData->numDocs); 
	for (int m = 0; m < pnewData->numDocs; m++)
	    new_nd[m] = 0;

	new_ndl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++)
	{
		new_ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
		    new_ndl[m][l] = 0;
	}

	new_ndlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++)
	{
		new_ndlz[m].resize(numSentiLabs);
	    for (int l = 0; l < numSentiLabs; l++)
		{
			new_ndlz[m][l].resize(numTopics);
			for (int z = 0; z < numTopics; z++)
			    new_ndlz[m][l][z] = 0; 
		}
	}

	new_nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
	{
		new_nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
		{
			new_nlzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++)
			    new_nlzw[l][z][r] = 0;
		}
	}

	new_nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
	{
		new_nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
		    new_nlz[l][z] = 0;
	}


	// model parameters
	newpi_dl.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++)
		newpi_dl[m].resize(numSentiLabs);

	newtheta_dlz.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++)
	{
		newtheta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++)
			newtheta_dlz[m][l].resize(numTopics);
	}

	newphi_lzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++)
	{
		newphi_lzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++)
			newphi_lzw[l][z].resize(pnewData->vocabSize);
	}

	// hyperparameters
	// alpha
	_alpha =  (double)pnewData->aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);

	alpha_lz.resize(numSentiLabs);
	alphaSum_l.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		alphaSum_l[l] = 0.0;
		alpha_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			alpha_lz[l][z] = _alpha;
			alphaSum_l[l] += alpha_lz[l][z];
		}
	}

	// gamma
	gamma_l.resize(numSentiLabs);
	gammaSum = 0.0;
	for (int l = 0; l < numSentiLabs; l++) {
		gamma_l[l] = (double)pnewData->aveDocLength * 0.05 / (double)numSentiLabs;
		gammaSum += gamma_l[l];
	}

	//beta
	if (_beta <= 0) _beta = 0.01;

	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			beta_lzw[l][z].resize(pnewData->vocabSize);
			for (int r = 0; r < pnewData->vocabSize; r++) {
			//beta_lzw[l][z].resize(vocabSize);
			//for (int r = 0; r < vocabSize; r++) 
				beta_lzw[l][z][r] = _beta; 
				betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		} 		
	}

	/*if (sentiLexFile != "") {
	// word prior transformation matrix
	lambda_lw.resize(numSentiLabs); 
	for (int l = 0; l < numSentiLabs; l++) {
	  lambda_lw[l].resize(pnewData->vocabSize);
		for (int r = 0; r < pnewData->vocabSize; r++)
			lambda_lw[l][r] = 1; 	
	}
	// MUST init beta_lzw first before incorporating prior information into beta
	prior2beta(); 

	}*/
	
	return 0;
}


int Inference::inf_sampling(int m, int n, int& sentiLab, int& topic)
{
	sentiLab = new_l[m][n];
	topic = new_z[m][n];
	int w = pnewData->pdocs[m]->words[n];   // word index of previous trained model
	int _w = pnewData->_pdocs[m]->words[n]; // word index of test data
	double u;
	
	new_nd[m]--;
	new_ndl[m][sentiLab]--;
	new_ndlz[m][sentiLab][topic]--;
	new_nlzw[sentiLab][topic][_w]--;
	new_nlz[sentiLab][topic]--;

    // do multinomial sampling via cumulative method, and p_st[l][k] is the temp variable for sampling
    for (int l = 0; l < numSentiLabs; l++) 
	{
  	    for (int k = 0; k < numTopics; k++) 
		{
		    //new_p[l][k] = (nlzw[l][k][w] + new_nlzw[l][k][_w] + beta_lzw[l][k][w]) / (nlz[l][k] + new_nlz[l][k] + betaSum_lz[l][k]) *
			 //    		(new_ndlz[m][l][k] + alpha_lz[l][k]) / (new_ndl[m][l] + alphaSum_l[l]) *
				//		(new_ndl[m][l] + gamma_l[l]) / (new_nd[m] + gammaSum);
		    new_p[l][k] = (nlzw[l][k][w] + new_nlzw[l][k][_w] + beta_lzw[l][k][_w]) / (nlz[l][k] + new_nlz[l][k] + betaSum_lz[l][k]) *
			     		(new_ndlz[m][l][k] + alpha_lz[l][k]) / (new_ndl[m][l] + alphaSum_l[l]) *
						(new_ndl[m][l] + gamma_l[l]) / (new_nd[m] + gammaSum);
		}
	}
	// cumulate multinomial parameters
	for (int l = 0; l < numSentiLabs; l++)      //sentiment label l must be > 1;  
	{    
		for (int k = 0; k < numTopics; k++) 
		{
			if (k==0)     // the first element of an sub array
			{
			    if (l==0) continue;
		        else new_p[l][k] += new_p[l-1][numTopics-1]; // accumulate the sum of the previous array
			}
			else new_p[l][k] += new_p[l][k-1];
	    }
	}
	// scaled sample because of unnormalized p_st[] ***--The normalization is been done here!!!--***
	u = ((double)rand() / RAND_MAX) * new_p[numSentiLabs-1][numTopics-1];

	// here we get the sample of label, from [0, S-1]
	for (sentiLab = 0; sentiLab < numSentiLabs; sentiLab++)
	{   for (topic = 0; topic < numTopics; topic++)
		{ 
		    if (new_p[sentiLab][topic] > u) 
		    goto stop;
		}
	}
    
	stop:
	if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1; // the max value of label is (S - 1)!!!
	if (topic == numTopics) topic = numTopics - 1; 


	// add newly estimated z_i to count variables
	new_nd[m]++;
	new_ndl[m][sentiLab]++;
	new_ndlz[m][sentiLab][topic]++;
	new_nlzw[sentiLab][topic][_w]++;
	new_nlz[sentiLab][topic]++;

    return 0;  
}


int Inference::read_newData(string filename)
{
	mapword2id::iterator it;
  map<int, int>::iterator _it;
	mapword2atr::iterator itatr; 
	mapword2prior::iterator sentiIt; 
	string line;
	char buff[BUFF_SIZE_LONG];
	size_t found;
	
    pnewData = new dataset;

	// read wordmap 
	string epochID = pmodelData->convertInt(pmodelData->epochID);
	pmodelData->read_wordmap(model_dir + "epoch_" + epochID + "_wordmap.txt", word2id);  // map word2id
  pmodelData->read_wordmap(model_dir + "epoch_" + epochID + "_wordmap.txt", id2word);  // mpa id2word
	
	// read sentiment lexicon file
	if (sentiLexFile != "") {
		if (pnewData->read_senti_lexicon((sentiLexFile).c_str())) {
			printf("Error! Can not open sentiFile %s to read!\n", sentiLexFile.c_str());
			delete pnewData;
			return 1;  // read sentiLex fail!
		}
		else 
			sentiLex = pnewData->sentiLex; 
	}
	
	//read_wordmap(wordmapfile, &word2id);
  if (word2id.size() <= 0) 
	{
	    printf("No word map available!\n");
	    return 1;
  }

	ifstream fin;
	fin.open(filename.c_str(), ifstream::in);

        if(!fin) {
	    printf("Cannot open file %s to read!\n", filename.c_str());
	    return 1;
  	}   

	vector<string> docs;  // documents at epoch t
	int numDocs = 0;
	
	while(!fin.getline(buff, BUFF_SIZE_LONG).eof()) {
		line = buff;
		if (line.find("[epoch_") == 0 ) {
			if (numDocs > 0) {
				break;
			}
		}
		else {
			docs.push_back(line);
			numDocs++;
		}
	}  // End while()

	fin.close();
	
	if (numDocs <= 0)
	{
		printf("Error! no documents found in file %s.\n", filename.c_str());
		return 1; 
	}
	
	pnewData->numDocs = numDocs;
 
    
  // allocate memory for corpus
  if (pnewData->pdocs) 
		pnewData->deallocate();
	else 
		pnewData->pdocs = new document*[pnewData->numDocs];
    pnewData->_pdocs = new document*[pnewData->numDocs];
    
  // set number of words to zero
	pnewData->vocabSize = 0;
	pnewData->epochSize = 0;
    
	for (int i = 0; i < pnewData->numDocs; i++) 
	{
		line = docs.at(i);	  
		strtokenizer strtok(line, " \t\r\n"); // \t\r\n are the separators of the words
		int docLength = strtok.count_tokens(); // return the length of document_i
		
		if (docLength <= 0) 
		{
			printf("Invalid (empty) document!\n");
			pnewData->deallocate();
			pnewData->numDocs = 0;
			pnewData->vocabSize = 0;
			
			return 1;
		}

	  pnewData->epochSize += docLength - 2; // i.e., do not include the document ID and time;
	
		vector<int> doc;
	  vector<int> _doc;
	  vector<int> priorSentiLabels;
	  
	  // generate ID for the vocabulary of the corpus, and assign each word token with the corresponding vocabulary ID. 
	  for (int k = 2; k < docLength; k++) 
		{	   
			it = word2id.find(strtok.token(k));
			if (it == word2id.end()) 
				pnewData->newWords.push_back(strtok.token(k).c_str());
			  // word not found, i.e., word unseen in training data
			  // do anything? (future decision)
			else 
			{
				int _id;
				_it = id2_id.find(it->second);
				if (_it == id2_id.end()) 
				{
				    _id = id2_id.size();
				    id2_id.insert(pair<int, int>(it->second, _id));
				    _id2id.insert(pair<int, int>(_id, it->second));
				} 
				else 
				    _id = _it->second;
		
				doc.push_back(it->second);
				_doc.push_back(_id);
			
				itatr = word2atr.find(strtok.token(k).c_str()); // match the token(j) to ID
				int priorSenti = -1;
				
				if (itatr == word2atr.end()) 
				{ // word not found, i.e., new word
					found = strtok.token(k).find('_');
	
					if (found!=string::npos) // word phrases
						priorSenti = pnewData->phrase_sentiment(strtok.token(k));
					else { // unigram
						sentiIt = sentiLex.find(strtok.token(k).c_str()); // check whether the word token can be found in the sentiment lexicon
						if (sentiIt != sentiLex.end())
							priorSenti = sentiIt->second.id;
					}
					
					// insert sentiment info into word2atr
					Word_atr temp = {_id, 1, priorSenti, false};  // vocabulary index; freqCount; word polarity; passFilterFlag 
					word2atr.insert(pair<string, Word_atr>(strtok.token(k), temp));
					priorSentiLabels.push_back(priorSenti);
				}
				
				else {
					priorSentiLabels.push_back(itatr->second.polarity);	
				}
			
			}
		}
		
		// allocate memory for new doc
		document * pdoc = new document(doc, priorSentiLabels, "inference");
		document * _pdoc = new document(_doc, priorSentiLabels, "inference");
		
		pdoc->docID = strtok.token(0).c_str();
		pdoc->timeID = strtok.token(1).c_str();
		_pdoc->docID = strtok.token(0).c_str();
		_pdoc->timeID = strtok.token(1).c_str();
		
		// add new doc
		pnewData->add_doc(pdoc, i);
		pnewData->_add_doc(_pdoc, i);
	}
    
    // update number of new words
	pnewData->vocabSize = id2_id.size();
	pnewData->aveDocLength = pnewData->epochSize / pnewData->numDocs;

	this->newNumDocs = pnewData->numDocs;
	this->newVocabSize = pnewData->vocabSize;
    if (newVocabSize == 0)
	{
	    printf("ERROR! Vocabulary size of the new dataset after removing new words is 0! \n");
		return 1; 
	}
	
	return 0;
}


void Inference::compute_newpi()
{
	for (int m = 0; m < pnewData->numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++)    
		    newpi_dl[m][l] = (new_ndl[m][l] + gamma_l[l]) / (new_nd[m] + gammaSum);
	}

}


void Inference::compute_newtheta()
{

	for (int m = 0; m < pnewData->numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++)  {
			for (int z = 0; z < numTopics; z++)
			    newtheta_dlz[m][l][z] = (new_ndlz[m][l][z] + alpha_lz[l][z]) / (new_ndl[m][l] + alphaSum_l[l]);    
		}
	}


}


int Inference::compute_newphi()
{
	map<int, int>::iterator it;

	for (int l = 0; l < numSentiLabs; l++)  {
	    for (int z = 0; z < numTopics; z++) {
			for(int r = 0; r < pnewData->vocabSize; r++) {
			    it = _id2id.find(r);
				if (it != _id2id.end())
				    //newphi_lzw[l][z][r] = (nlzw[l][z][it->second] + new_nlzw[l][z][r] + beta_lzw[l][z][it->second]) / (nlz[l][z] + new_nlz[l][z] + betaSum_lz[l][z]);
				    newphi_lzw[l][z][r] = (nlzw[l][z][it->second] + new_nlzw[l][z][r] + beta_lzw[l][z][r]) / (nlz[l][z] + new_nlz[l][z] + betaSum_lz[l][z]);
				else {
				    printf("Error! Can't find word [%d] !\n", r);
					return 1; 
				}
			}
		}
	}

	return 0;
}


int Inference::save_model(string model_name)
{
	/*if (save_model_newtassign(result_dir + model_name + tassign_suffix)) 
		return 1; */
	
	if (save_model_newtwords(result_dir + model_name + twords_suffix)) 
		return 1;

	if (save_model_newpi_dl(result_dir + model_name + pi_suffix)) 
		return 1;

	/*if (save_model_newtheta_dlz(result_dir + model_name + theta_suffix)) 
		return 1;

	if (save_model_newphi_lzw(result_dir + model_name + phi_suffix)) 
		return 1; */

	if (save_model_newothers(result_dir + model_name + others_suffix)) 
		return 1;

	//string epochID = pnewData->convertInt(this->epochID);
	//if (save_model_perplexity(result_dir + "epoch_" + epochID + ".perplexity")) 
	if (save_model_perplexity(result_dir + this->model_name + ".perplexity")) 
		return 1;

	return 0;
}


int Inference::save_model_perplexity(string filename)
{
	int currentStep = liter/savestep - 1; // current saving step 
	//bool fileExists;
	double perplexity; 
	FILE * fout;

	if (currentStep == 0) { // if this is the first time writing the file, remove the existing result files
		fout = fopen(filename.c_str(), "w"); 
		if (!fout) { 
			printf("Cannot open file %s to write!\n", filename.c_str());
			return 1;
		}
		fprintf(fout, "Iteration    Perplexity\n");
	}
	else {
		fout = fopen(filename.c_str(), "a"); // use 'a' for append operation!!!!
		if (!fout) {
			printf("Cannot open file %s to save!\n", filename.c_str());
			return 1;
		}
	}
	
	perplexity = compute_perplexity();
	fprintf(fout, "%9d    %f \n", liter, perplexity);

	fclose(fout);

	return 0;
}


int Inference::save_model_newpi_dl(string filename)
{

    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
    }

	for (int m = 0; m < pnewData->numDocs; m++) 
	{ 
		fprintf(fout, "d_%d %s ", m, pnewData->pdocs[m]->docID.c_str());
		for (int l = 0; l < numSentiLabs; l++) 
			fprintf(fout, "%f ", newpi_dl[m][l]);
	    
		fprintf(fout, "\n");
    }
   
    fclose(fout);       
	return 0;
}



int Inference::save_model_newtheta_dlz(string filename)
{
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
		printf("Cannot open file %s to save!\n", filename.c_str());
		return 1;
    }
    
    for(int m = 0; m < pnewData->numDocs; m++) 
    {
        fprintf(fout, "Document %d\n", m);
	    for (int l = 0; l < numSentiLabs; l++) 
	    { 
	        for (int z = 0; z < numTopics; z++) 
		        fprintf(fout, "%f ", newtheta_dlz[m][l][z]);
		    
		    fprintf(fout, "\n");
		 }
    }

    fclose(fout);
	return 0;
}


int Inference::save_model_newphi_lzw(string filename)
{

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
	for (int l = 0; l < numSentiLabs; l++)
	{  
	    for (int z = 0; z < numTopics; z++) 
		{ 
		    fprintf(fout, "Label:%d  Topic:%d\n", l, z);
     	    for (int r = 0; r < pnewData->vocabSize; r++) 
			    fprintf(fout, "%.15f ", newphi_lzw[l][z][r]);
	        
            fprintf(fout, "\n");
	   }
    }
    
    fclose(fout);    
	return 0;
}


int Inference::save_model_newothers(string filename)
{
	
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
	fprintf(fout, "model_dir=%s\n", model_dir.c_str());
	fprintf(fout, "data_dir=%s\n", data_dir.c_str());
	fprintf(fout, "result_dir=%s\n", result_dir.c_str());
	fprintf(fout, "datasetFile=%s\n", datasetFile.c_str());
	fprintf(fout, "model_name=%s\n", model_name.c_str());
	//fprintf(fout, "muFile=%s\n", muFile.c_str());
	//fprintf(fout, "epsilonFile=%s\n", epsilonFile.c_str());
	fprintf(fout, "betaFile=%s\n", betaFile.c_str());

	fprintf(fout, "\n");
	fprintf(fout, "Test document %s statistics: \n", datasetFile.c_str());
	fprintf(fout, "Loaded model epochID=%d\n", epochID);
    fprintf(fout, "numDocs=%d\n", pnewData->numDocs);
	fprintf(fout, "epochSize=%d\n", pnewData->epochSize);
	fprintf(fout, "numNewWords=%d\n", (int)(pnewData->newWords.size()));
	fprintf(fout, "aveDocLength=%d\n", pnewData->aveDocLength);
    fprintf(fout, "newVocabSize=%d\n", pnewData->vocabSize);
	fprintf(fout, "numSentiLabs=%d\n", numSentiLabs);
	fprintf(fout, "numTopics=%d\n", numTopics);
	fprintf(fout, "liter=%d\n", liter);
	fprintf(fout, "savestep=%d\n", savestep);
	
	for (int l = 0; l < numSentiLabs; l++) 
	{
	    for (int z = 0; z < numTopics; z++) 
		{
		    if (z == 0) fprintf(fout, "alpha[%d]={", l);
		    fprintf(fout, "%.2g ", alpha_lz[l][z]);
		}
		fprintf(fout, "}\n");
	}


	for (int l = 0; l < numSentiLabs; l++) 
	{
		fprintf(fout, "alphaSum[%d]=%g\n", l, alphaSum_l[l]);
	}


	// gamma 
	for (int l = 0; l < numSentiLabs; l++) 
	{
		fprintf(fout, "gamma[%d]=%g\n", l, gamma_l[l]);
	}


	// beta
	//fprintf(fout, "_beta[0]=%g\n", _beta);
	for (int l = 0; l < numSentiLabs; l++)
	{
	    for (int z = 0; z < numTopics; z++) // Only print out 3 topics
		{
		    fprintf(fout, "betaSum[%d][%d]=%f\n", l, z, betaSum_lz[l][z]);
		}
	}
	
	 fclose(fout);      

	return 0;
}



int Inference::save_model_newtwords(string filename)
{

	mapid2word::iterator it; // typedef map<int, string> mapid2word, using map class
	map<int, int>::iterator _it;

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }
    
    if (twords > pnewData->vocabSize) 
	{    // equivalent to print out the whole vacabulary list
	    twords = pnewData->vocabSize;
    }
   
   
    for (int l = 0; l < numSentiLabs; l++)
    { 
        fprintf(fout, "Label %dth\n", l);
        for (int k = 0; k < numTopics; k++) 
	    { 
	        vector<pair<int, double> > words_probs;  // "words_probs"!!! what is the data structure of "words_probs"?
	        pair<int, double> word_prob;             // "word_prob"!!! different from above!!!
	        for (int w = 0; w < pnewData->vocabSize; w++) 
	        { 
		        word_prob.first = w; // w is the new word id 
	            word_prob.second = newphi_lzw[l][k][w]; // the topic-word probability
	            words_probs.push_back(word_prob);
	        }
    
            // quick sort to sort word-topic probability
		    std::sort(words_probs.begin(), words_probs.end(), sort_pred());

	        fprintf(fout, "Topic %dth:\n", k);
	        for (int i = 0; i < twords; i++) 
	        { 
				_it = _id2id.find(words_probs[i].first);
				if (_it == _id2id.end()) 
				{
		            continue;
	            }

				it = id2word.find(_it->second);
	            if (it != id2word.end()) 
		        { 
			        fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
	            } 
	        }
	    } // for topic
    } // for label
     
    fclose(fout);      
	return 0;
}


int Inference::save_model_newtassign(string filename)
{
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) 
	{
	    printf("Cannot open file %s to save!\n", filename.c_str());
	    return 1;
    }

    // wirte docs with topic assignments for words
	for (int m = 0; m < pnewData->numDocs; m++) 
	{    
		fprintf(fout, "%s \n", pnewData->pdocs[m]->docID.c_str());
		for (int n = 0; n < pnewData->pdocs[m]->length; n++) 
		{
	        fprintf(fout, "%d:%d:%d ", pnewData->pdocs[m]->words[n], new_l[m][n], new_z[m][n]); //  wordID:sentiLab:topic
	    }
	    fprintf(fout, "\n");
    }

    fclose(fout);

	return 0;
}



double Inference::compute_perplexity()
{ 
	int _ndSum = 0;  // _ndSum = \sum_d[n_test_d]
	//double p_m = 0.0;  // p_w = \pro[\sum_l \sum_z[p(w|l,z)*p(z|l)*p(l)]]
	vector<long double> prob_phi_theta_pi;
    double perplexity = 0.0;
	double numerator = 0.0;
	
	// compute _ndSum
	for (int m = 0; m < pnewData->numDocs; m++)
		_ndSum += pnewData->pdocs[m]->length;

	// compute p_m, i.e., the perplexity contribution of document m 
	prob_phi_theta_pi.resize(pnewData->numDocs);
	for (int m = 0; m < pnewData->numDocs; m++)
	    prob_phi_theta_pi[m] = 0.0; // 1.0
	
	long double p_w = 0.0;
	for (int m = 0; m < pnewData->numDocs; m++) {
		for (int n = 0; n < pnewData->_pdocs[m]->length; n++) {
			for (int l = 0; l < numSentiLabs; l++) {
				for (int z = 0; z < numTopics; z++)
					p_w += newphi_lzw[l][z][pnewData->_pdocs[m]->words[n]] * newtheta_dlz[m][l][z] * newpi_dl[m][l];
			}
			//prob_phi_theta_pi[m] = prob_phi_theta_pi[m] * p_w;  // **** if prob_phi_theta_pi == 0 -- some special treatment
			prob_phi_theta_pi[m] += log(p_w);
			p_w = 0.0;
		}
	}
	//fclose(fout);

	for (int m = 0; m < pnewData->numDocs; m++)
		numerator += prob_phi_theta_pi[m];
		
	perplexity = exp(-(double)numerator / _ndSum);

	return (perplexity);
}


