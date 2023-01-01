
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

#ifndef	_DATASET_H
#define	_DATASET_H

//#include "model.h"
#include "constants.h"
#include "document.h"
#include "map_type.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

using namespace std; 

//struct Word_atr { 
//	int id; // vocabulary index
//	int freqCount;
//	int polarity;
//	bool passFilterFlag; // initially set to false 
//};
//
//
//// map of words/terms [string => int]
//typedef map<string, int> mapword2id;
//// map of words/terms [int => string]
//typedef map<int, string> mapid2word;
//// map of words/attributes_of_words [string => word_attr]
//typedef map<string, Word_atr> mapword2atr;



class dataset
{
public:
	mapword2atr word2atr; // word2atr is a universal variable for all the epoch {0 ... T}
	mapid2word id2word; 
	mapword2prior sentiLex; // <string, int> => <word, polarity>
	
	//mapword2id epochSentiWords; // only store information for CURRENT epoch! --  total number of pos terms in the sentiment lexicon that has appear in the corpus (not instances)
	
	document ** pdocs; // store the old word ID
	document ** _pdocs; // only use for inference, i.e., for storing the new word ID 

	ifstream fin;
	bool buffEpochFinish_flag;
	
	// command line argument parse
	string data_dir;
	string result_dir;
	string wordmapfile;

	/*int niters;
	int savestep;
	int twords;
	int updateParaStep;
	double _alpha;
	double _beta;
	double _gamma[3];*/

	// corpus parameter
	int corpusVocabSize;
	mapword2id corpusVocab;   // <string, int> ==> <word, corpusWideVocabID>
	map<int, int> epochVocabID2corpusVocabID;
	map<int, int> corpusVocabID2epochVocabID;
	
	// epoch t specific dataset parameters 
	int numDocs; // number of documents in epoch t
	int aveDocLength; // average document length in epoch t
	int totalNumDocs; // total number of documents processed in the file stream 
	int vocabSize; // number of vocabulary terms -- this should be corpus-wise??
	int epochSize; // total number of word tokens of the documents in epoch t 
	int epochID; 
	
	
	vector<string> docs; // for buffering the documents of epoch t
	vector<string> newWords;
	
	
	// sentiment words statistics
	int numPosWordLex; // total number of pos words in the sentiment lexicon
	int numNegWordLex;
	int numNeuWordLex; 
	int numPosWordCorpus; // total number of postive words in the sentiment lexicon that match the corpus 
	int numNegWordCorpus;
	int numNeuWordCorpus;
		
	

	// functions 
	dataset(void);
	dataset(string result_dir, int epochID, mapword2id corpusVocab, mapword2prior sentiLex);
	~dataset(void);
	
	int read_dataStream(ifstream& fin);
	int read_senti_lexicon(string sentiLexiconFileDir);
	
	//int read_infData(string filename);
	int analyzeEpoch(vector<string>& docs);


	static int write_wordmap(string wordmapfile, mapword2atr& pword2atr);
	static int read_wordmap(string wordmapfile, mapid2word& pid2word);
	static int read_wordmap(string wordmapfile, mapword2id& pword2id); 
	
	//int update_expcted_ntlzw(); // i.e., _ntlzw. Update after processing each epoch. 
	int init_parameter();
	//int update_parameter(); // update epsilon_slzw_old and _ntlzw due to new words appear in the current epoch
	bool loadEpochStatus(int& timeSpam, int sliceLength);
	void deallocate();  
	void add_doc(document * doc, int idx);
	void _add_doc(document * doc, int idx);

	//void print_dataset_statistics();
	void print_epoch_statistics();
	void print_sentiLexStatistics();
	void print_matrix();

	string convertInt(int number);
	int phrase_sentiment(string phrase);

};

#endif
