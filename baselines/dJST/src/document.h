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

   Author: 	Chenghua Lin, c.lin@open.ac.uk 
   					Revised by Yulan He, 06/05/2011	
*/
   
   
#ifndef	_DOCUMENT_H
#define	_DOCUMENT_H

#include <vector>
#include <iostream>

using namespace std; 



class document {
public:
	int * words;
	int * priorSentiLabels;
	string docID;
	string timeID;
	string rawstr;
	int length;
	
	
	document() {
		words = NULL;
		priorSentiLabels = NULL;
		docID = "";
		timeID = "";
		rawstr = "";
		length = 0;	
	}
    
    document(int length) { // Constructor. Retrieve the length of the document and allocate memory for storing the documents
		this->length = length;
		docID = "";
		timeID = "";
		rawstr = "";
		words = new int[length]; // words stores the word token ID, which is integer
		priorSentiLabels = new int[length];	
    }
    
    document(int length, int * words) { // Constructor. Retrieve the length of the document and store the element of words into the array
		this->length = length;
		docID = "";
		timeID = "";
		rawstr = "";
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = words[i];
		}
		priorSentiLabels = new int[length];	
    }

    document(int length, int * words, string rawstr) { // Constructor. 
		this->length = length;
		docID = "";
		timeID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			 this->words[i] = words[i];
		}
		priorSentiLabels = new int[length];	
    }
    

    document(vector<int> & doc) {
		this->length = doc.size();
		docID = "";
		timeID = "";
		rawstr = "";
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
		}
		priorSentiLabels = new int[length];	
    }


	document(vector<int> & doc, string rawstr) {
		this->length = doc.size();
		docID = "";
		timeID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
		}
		priorSentiLabels = new int[length];	
	}

    document(vector<int> & doc, vector<int> &priorSentiLab, string rawstr) {
		this->length = doc.size();
		docID = "";
		timeID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		this->priorSentiLabels = new int[length];	
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
			this->priorSentiLabels[i] = priorSentiLab[i];	
		}
    }
    
    ~document() {
		if (words != NULL){ 
			delete [] words;
			words = NULL;
		}
			
		if (priorSentiLabels != NULL){
			delete [] priorSentiLabels;
			priorSentiLabels = NULL;
		}
    }
};

#endif
