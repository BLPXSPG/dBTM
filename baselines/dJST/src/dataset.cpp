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
	   Revised by Yulan He, 06/06/2011
*/

#include "dataset.h"
#include "document.h"
#include "model.h"
#include "map_type.h"
#include "strtokenizer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace std; 

dataset::dataset(void)
{
	pdocs = NULL;
	_pdocs = NULL;
	word2atr.clear();
	
	// command line arguement parse
	data_dir = ".";
	result_dir = ".";
	//datasetFile = "reviews.dat";
	//sentiLexFile = "mpqaApp.lex";
	wordmapfile = "_wordmap.txt";
	
	//corpus parameters
	corpusVocabSize = 0;
	corpusVocab.clear();
	
	// epoch t specific dataset parameters 
	numDocs = 0; 
	aveDocLength = 0; 
	totalNumDocs = 0;   
	vocabSize = 0; 
	epochSize = 0; 
	epochID = 0;
	
	numPosWordLex = 0; 
	numNegWordLex = 0;
	numNeuWordLex = 0; 
	numPosWordCorpus = 0;
	numNegWordCorpus = 0;
	numNeuWordCorpus = 0;
}


dataset::dataset(string result_dir, int epochID, mapword2id corpusVocab, mapword2prior sentiLex)
{
	pdocs = NULL;
	_pdocs = NULL;
	this-> corpusVocab = corpusVocab;
	this->sentiLex = sentiLex;
	
	// command line arguement parse
	data_dir = ".";
	this->result_dir = result_dir;
	wordmapfile = "_wordmap.txt";
	
	//corpus parameters
	epochVocabID2corpusVocabID.clear();
	corpusVocabID2epochVocabID.clear();
	
	// epoch t specific dataset parameters 
	word2atr.clear();
	numDocs = 0; 
	aveDocLength = 0; 
	totalNumDocs = 0;   
	vocabSize = 0; 
	epochSize = 0; 
	this->epochID = epochID;
	
	numPosWordLex = 0; 
	numNegWordLex = 0;
	numNeuWordLex = 0; 
	numPosWordCorpus = 0;
	numNegWordCorpus = 0;
	numNeuWordCorpus = 0;
}


dataset::~dataset(void)
{
	deallocate();
}


int dataset::read_dataStream(ifstream &fin)
{
	string line;
	char buff[BUFF_SIZE_LONG];

	docs.clear();  // documents at epoch t
	numDocs = 0;
		
	while (!fin.getline(buff, BUFF_SIZE_LONG).eof()) {
		line = buff;
		if (line.find("[epoch_") != 0) {
			docs.push_back(line);
			numDocs++;
		}
	}  // End while()
	
	// epoch buffering finished -- start analyzing the epoch
	if (numDocs > 0) {
		analyzeEpoch(docs);
	}
	
	return 0;
}


int dataset::analyzeEpoch(vector<string>& docs)
{ 
	mapword2atr::iterator it; 
	mapword2id::iterator vocabIt;   
	mapword2prior::iterator sentiIt;
	map<int,int>::iterator idIt;
		
	string line;
	numDocs = docs.size(); // number of documents at epoch t
	vocabSize = 0;  // reset values for each new epoch 
	epochSize = 0;
	aveDocLength = 0; 
	newWords.clear();
	bool newWordFound_flag = false; 
	size_t found;
	int corpusVocabID;

  // allocate memory for corpus, where docs is an instance of the document Class defined within dataset Class
  if (pdocs) {
		printf("Warning! Memory of variable 'pdocs' in previous epoch has not been released!\n");
		deallocate();
		pdocs = new document*[numDocs];
    } 
	else 
		pdocs = new document*[numDocs]; // the 'docs' here is the document Class type array pointer which points to all M documents
	
	for (int i = 0; i < (int)docs.size(); ++i) {			
		line = docs.at(i);
		strtokenizer strtok(line, " \t\r\n"); // \t\r\n are the separators of the words
		int docLength = strtok.count_tokens(); // return the length of document_i
	
		if (docLength <= 0) {
			printf("Invalid (empty) document!\n");
			deallocate();
			numDocs = vocabSize = 0;
			return 1;
		}
	
		epochSize += docLength - 2; 
		
		// allocate memory for the new document_i 
		document * pdoc = new document(docLength-2);
		pdoc->docID = strtok.token(0).c_str();
		pdoc->timeID = strtok.token(1).c_str();

		// generate ID for the vocabulary of the corpus, and assign each word token with the corresponding vocabulary ID. 
		for (int k = 0; k < docLength-2; k++) {			
			int priorSenti = -1;	
			it = word2atr.find(strtok.token(k+2).c_str()); // match the token(j) to ID
		
			if (it == word2atr.end()) { //  i.e., new word
				pdoc->words[k] = word2atr.size(); // the ID of the new word is equal to the largest ID in the word2id + 1, as the smallest ID is 0
				// determine whether the token is phrase or unigram 
				found = strtok.token(k+2).find('_');
	
				if (found!=string::npos) // word phrases
					priorSenti = phrase_sentiment(strtok.token(k+2));
				else { // unigram
					sentiIt = sentiLex.find(strtok.token(k+2).c_str()); // check whether the word token can be found in the sentiment lexicon
					if (sentiIt != sentiLex.end())
						priorSenti = sentiIt->second.id;
				}
					
				// insert sentiment info into word2atr
				Word_atr temp = {word2atr.size(), 1, priorSenti, false};  // vocabulary index; freqCount; word polarity; passFilterFlag 
				word2atr.insert(pair<string, Word_atr>(strtok.token(k+2), temp));
				pdoc->priorSentiLabels[k] = priorSenti;
				
			} 
		       	
			else { // word already stacked in word2atr 
				pdoc->words[k] = it->second.id;
				pdoc->priorSentiLabels[k] = it->second.polarity;	 // initialize all the word token sentiment flag
			}
			
			vocabIt = corpusVocab.find(strtok.token(k+2).c_str()); // match the token(j) to ID
		
			if (vocabIt == corpusVocab.end()) { //  i.e., word first encountered in the whole corpus
				if (epochID > 0) {
					newWordFound_flag = true; 
					newWords.push_back(strtok.token(k+2).c_str());
				}
				corpusVocabID = corpusVocab.size(); // the ID of the new word is equal to the largest ID in the word2id + 1, as the smallest ID is 0
				corpusVocab.insert(pair<string, int>(strtok.token(k+2), corpusVocabID));
			}
			else
				corpusVocabID = vocabIt->second;

			corpusVocabID2epochVocabID.insert(pair<int, int>(corpusVocabID, pdoc->words[k]));	
			epochVocabID2corpusVocabID.insert(pair<int, int>(pdoc->words[k], corpusVocabID));
		  
		} // End: for (int k = 0; k < docLength; k++) 
		
		add_doc(pdoc, i); // attach new doc to the corpus
	} 
	    
	    
	// update number of words
	vocabSize = word2atr.size();
	corpusVocabSize = corpusVocab.size();
	
	// calcualte average document length in epoch t
	aveDocLength = epochSize/numDocs;
	
	// ************************
	if (newWordFound_flag) {
		// update the evolutionary parameter
		// ********** NEED to work on here again!!	  dataset::update_parameter();
		
		printf("%d new words found in <epoch_%d>:\n", (int)newWords.size(), epochID);
		for (int j = 0; j < (int)newWords.size(); j++) 
			printf("%s\n", newWords.at(j).c_str());
	}

    if (write_wordmap(result_dir + "epoch_" + convertInt(epochID) + wordmapfile, word2atr)) {
		printf("ERROE! Can not read wordmap file %s!\n", wordmapfile.c_str());
		return 1; 
  	}

	if (read_wordmap(result_dir + "epoch_" + convertInt(epochID) + wordmapfile, id2word)) {
		printf("ERROE! Can not read wordmap file %s!\n", wordmapfile.c_str());
		return 1; 
	}

	// clear vector docs for the new documents of next epoch
	docs.clear();

	return 0;
}


void dataset::print_epoch_statistics()
{
	printf("numDocs = %d\n", numDocs);
	printf("vocabSize = %d\n", vocabSize);
	printf("epochSize = %d\n", epochSize);
	printf("aveDocLength = %d\n", aveDocLength);
}


void dataset::print_sentiLexStatistics()
{
	//printf("Lexicon file %s statistics:\n", sentiLexFile.c_str());
	printf("numPosWord = %d, numNegWord = %d, numNeuWord = %d\n", numPosWordLex, numNegWordLex, numNeuWordLex);
}



void dataset::deallocate() 
{
	if (pdocs) {
		for (int i = 0; i < numDocs; i++) 
			delete pdocs[i];		
		delete [] pdocs;
		pdocs = NULL;
	}
	
	if (_pdocs) {
		for (int i = 0; i < numDocs; i++) 
			delete _pdocs[i];
		delete [] _pdocs;
		_pdocs = NULL;
	}
}
    

void dataset::add_doc(document * doc, int idx) 
{
  if (0 <= idx && idx < numDocs) // why not 0 <= idx < M ???
	    pdocs[idx] = doc;
}   

void dataset::_add_doc(document * doc, int idx) 
{
    if (0 <= idx && idx < numDocs) 
	{   // why not 0 <= idx < M ???
	    _pdocs[idx] = doc;
    }
}


int dataset::read_senti_lexicon(string sentiLexiconFile) 
{
	sentiLex.clear();
	char buff[BUFF_SIZE_SHORT];
    string line;
    vector<double> wordPrior;
    int labID;
    double tmp, val;
    int numSentiLabs;
    
    FILE * fin = fopen(sentiLexiconFile.c_str(), "r");
    if (!fin) {
		printf("Cannot open file %s to read!\n", sentiLexiconFile.c_str());
		return 1;
    }    
     
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin) != NULL) {
		line = buff;
		
		strtokenizer strtok(line, " \t\r\n");
			
		if (strtok.count_tokens() < 1)  { // some word has 3 properties while others may have 4 
			printf("Warning! The strtok count in the lexicon line [%s] is smaller than 2!\n", line.c_str());
			//return 1;
		}
		else {	
			tmp = 0.0;
			labID = 0;
			wordPrior.clear();
			numSentiLabs = strtok.count_tokens();
			for (int k = 1; k < strtok.count_tokens(); k++) {
				val = atof(strtok.token(k).c_str());
				if (tmp < val) {
					tmp = val;
					labID = k-1;
				}
				wordPrior.push_back(val);
			}
			Word_Prior_Attr temp = {labID, wordPrior};  // sentiment label ID, sentiment label distribution
			sentiLex.insert(pair<string, Word_Prior_Attr >(strtok.token(0), temp));
		}
    }
    
	if (sentiLex.size() <= 0) {
		printf("Can not find any sentiment lexicon in file %s!\n", sentiLexiconFile.c_str());
		return 1;
	}

	// print sentiment lexicon file for debugging purpose
	/*mapword2prior::iterator sentiIt;
	for (sentiIt = sentiLex.begin(); sentiIt != sentiLex.end(); sentiIt++) {
		cout << sentiIt->first << " " << sentiIt->second.id << endl;
		for (int j = 0; j < numSentiLabs; j++)  {
			cout << sentiIt->second.labDist[j] << " ";
		}
		cout << endl;
	}*/
	
    fclose(fin);
    return 0;
}


int dataset::write_wordmap(string wordmapfile, mapword2atr &pword2atr) 
{
    FILE * fout = fopen(wordmapfile.c_str(), "w");
    if (!fout) {
		printf("Cannot open file %s to write!\n", wordmapfile.c_str());
		return 1;
    }    
    
    mapword2atr::iterator it;
    fprintf(fout, "%d\n", (int)(pword2atr.size()));
    for (it = pword2atr.begin(); it != pword2atr.end(); it++) {
	    fprintf(fout, "%s %d\n", (it->first).c_str(), it->second.id);
    }
    
    fclose(fout);
    
    return 0;
}


int dataset::read_wordmap(string wordmapfile, mapid2word &pid2word) 
{
    pid2word.clear(); 
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
		printf("Cannot open file %s to read!\n", wordmapfile.c_str());
		return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff); // assign nwords the total number of vacabulary, by reading the first line of "wordmap.txt"
    
    for (int i = 0; i < nwords; i++) {
		fgets(buff, BUFF_SIZE_SHORT - 1, fin);
		line = buff; // read each single vocabulary from "wordmap.txt"
		
		strtokenizer strtok(line, " \t\r\n"); // \t\r\n is the data data separator, so called white space
		if (strtok.count_tokens() != 2) {
			printf("Warning! Line %d in %s contains less than 2 words!\n", i+1, wordmapfile.c_str());
			continue;
		}
		
		pid2word.insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0)));
    }
    
    fclose(fin);
    
    return 0;
}


int dataset::read_wordmap(string wordmapfile, mapword2id& pword2id) 
{
    pword2id.clear();
    char buff[BUFF_SIZE_SHORT];
    string line;


    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
		printf("Cannot open file %s to read!\n", wordmapfile.c_str());
		return 1;
    }    
    
        
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
		fgets(buff, BUFF_SIZE_SHORT - 1, fin);
		line = buff;
		
		strtokenizer strtok(line, " \t\r\n");
		if (strtok.count_tokens() != 2)
			continue;
		
		pword2id.insert(pair<string, int>(strtok.token(0), atoi(strtok.token(1).c_str())));
    }
    
    fclose(fin);
    
    return 0;
}


string dataset::convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    
	return ss.str();//return a string with the contents of the stream
}


int dataset::phrase_sentiment(string phrase)
{
	int sentiLab;
	mapword2prior::iterator sentiIt;   
	vector<int> labs;
	bool firstFlag = true;
	int numSentiLabs = 0;
	
	strtokenizer strtok(phrase, "_"); // \t\r\n are the separators of the words
	
	for (int i = 0; i < strtok.count_tokens(); i++) {
	    sentiIt = sentiLex.find(strtok.token(i).c_str());
		
		if (sentiIt != sentiLex.end()) {
			if (firstFlag) {
				numSentiLabs = sentiIt->second.labDist.size();
				labs.resize(numSentiLabs);
				for (int i = 0; i < numSentiLabs; i++)
					labs[i] = 0;
				firstFlag = false;
			}
			labs[sentiIt->second.id]++;
		}
	}

	int tmp = 0;
	sentiLab = -1;
	for (int i = 0; i < numSentiLabs; i++) {
		if (tmp < labs[i]) {
			tmp = labs[i];
			sentiLab = i;
		}
	}

	return sentiLab;
}

//
//
//int dataset::read_infData(string filename)
//{
//	ifstream fin;
//	vector<string> docs_string; 
//	string line;
//	char buff[BUFF_SIZE_LONG];
//	map<int, int> id2_id;
//
//	
//	// read new document 
//	fin.open(filename.c_str(), ifstream::in);
//	
//	if(!fin)
//	{
//	    printf("Error! Can not open dataset %s to read!\n", filename.c_str());
//	    return 1; 
//	}
//
//
//	docs_string.clear(); // doccuments of epoch t 
//	line = buff;
//	strtokenizer strtok(line, " \t\r\n"); // \t\r\n are the separators of the words
//	int docLength = strtok.count_tokens(); // return the length of document_i
//
//	if (docLength != 2)
//	{
//		printf("Error! The first line of contains %d elments! Should only contain 2 elements.\n", docLength);
//	    return 1; 
//	}
//	else 
//	{
//		newNumDocs = atoi(strtok.token(1).c_str());  // read number of documents in current epoch
//		if (newNumDocs <= 0)
//		{
//			printf("Error! numDocs = %d\n", numDocs);
//	        return 1; 
//		}
//	}
//
//	// fetch all the documents for the epoch
//	for (int i = 0; i < newNumDocs; i++)
//	{
//		if (!fin.getline(buff, 10000).eof())
//		{
//			line = buff;
//			docs_string.push_back(line);
//		}
//		else 
//		{
//		    printf("Error! Could not read docuemnt_%d of epoch_\n", i);
//	        return 1; 
//		}
//	}
//
//
//	// read wordmap from previous model 
//	dataset::read_wordmap(wordmapfile, word2id);
//    if (word2id.size() <= 0) 
//	{
//	    printf("No word map available!\n");
//	    return 1;
//    }
//
//	// allocate memory for corpus
//    if (pmodelData->pdocs) 
//	{
//	    pmodelData->deallocate();
//    } 
//	else 
//	{
//	    pmodelData->pdocs = new document*[];
//    }
//    _docs = new document*[M];
//
//
//	return 0;
//}
