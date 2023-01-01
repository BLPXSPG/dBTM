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

#ifndef _UTILS_H
#define _UTILS_H

#include "dataset.h"
//#include "model_inf.h"
#include <string>
#include <algorithm>


using namespace std;

struct sort_pred  // for sorting word probability 
{ bool operator()(const std::pair<int,double> &left, const std::pair<int,double> &right) 
   {  return left.second > right.second; 
   }
};

//struct sort_high2low  // for sorting from high freq. to low freq. 
//{ bool operator()(const std::pair<string,int> &left, const std::pair<string,int> &right) 
//   {  return left.second > right.second; 
//   }
//};
//
//struct sort_low2high  // for sorting from low freq. to high freq. 
//{ bool operator()(const std::pair<string,int> &left, const std::pair<string,int> &right) 
//   {  return left.second < right.second; 
//   }
//};

class model;
class Inference;

class utils {
	private:
		int model_status;
		string mode_str;
    string model_dir;
		string data_dir;
		string result_dir;
    string model_name;
		string wordmapfile;
		string sentiLexFile;
		string docLabFile;
    string datasetFile;
    string configfile;
		string opt_mu_mode;
		int numScales; 
    int numSentiLabs;
		int numTopics;
    int niters;
    int savestep;
    int twords;
		int updateParaStep; 
		int epochLength;
		double alpha;
		double beta;
    double gamma;    // this is for the convience of parsing the command line arguements.
		bool useHistoricalAlpha;
	
	
public:
		// constructor
		utils();
		
    // parse command line arguments
    int parse_args(int argc, char ** argv, int&  model_status);
  	int parse_args(int argc, char ** argv, model * pmodel);
		int parse_args(int argc, char ** argv, Inference * pmodel_inf);
    
    // read configuration file
    int read_config_file(string configfile);
    
    // read and parse model parameters from <model_name>.others
    int read_and_parse(string filename, Inference * model_inf); 
  
    // generate the model name for the current iteration
    // iter = -1 => final model
    string generate_model_name(int iter, int epochID);  
    string generate_model_name(int iter);  
    
    string generate_result_name(string dfile, int numTopics, double alpha, double gamma[], int neuTH_L, int neuTH_H, int posTH_L, int posTH_H, int negTH_L, int negTH_H);
  
    // make directory
    int make_dir(string strPath);
    
    // sort    
    void sort(vector<double> & probs, vector<int> & words);
    //static void quicksort(vector<pair<int, double> > & vect, int left, int right);
};

#endif

