## Citation
Source code for the paper: Tracking Brand-Associated Polarity-Bearing Topics in User Reviews. Runcong Zhao, Lin Gui, Hanqi Yan and Yulan He (TACL).

## Environment
Dependency packages are in requirement.txt, you can use `pip` to install:

```{bash}
(envfordbtm)$ pip install -r requirements.txt
```

## Data
Preprocessed data included in [data/beauty_makeupalley/] can be used directly for dBTM, O-dBTM. It can also be used for baseline BTM, dJST and TBIP, with some tiny change to fit the input formats of those models.
The original data is from [MakeupAlley](https://www.makeupalley.com/), a review website on beauty products.
Another dataset used in the paper is [HotelRec](https://github.com/Diego999/HotelRec), a hotel recommendation dataset based on TripAdvisor.

To include a customized data set, first create a repo `data/{dataset_name}/time`. The following files must be inside this folder:

* `counts.npz`: a `[num_documents, num_words]` 
  [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) 
  containing the word counts for each document.
* `brand_indices.npy`: a `[num_documents]` vector where each entry is an
  integer in the set `{0, 1, ..., num_brands - 1}`, indicating the brand of 
  the corresponding document in `counts.npz`.
* `score_indices.npy`: a `[num_documents]` vector where each entry is an
  integer in the set `{-1, 0, 1}`, indicating the review polarity of 
  the corresponding document in `counts.npz`.

Also in `data/{dataset_name}/clean`. The following files must be inside this folder:
* `brand_map.txt`: a `[num_brands]`-length file where each line denotes the name of the brand in the corpus.
* `vocabulary.txt`: a `[num_words]`-length file where each line denotes the corresponding word in the vocabulary.

## Learning
Run dBTM.py with the command:
```{bash}
(envfordbtm)$ python setup/poisson_factorization_pretrain_t.py  --data=beauty_makeupalley
(envfordbtm)$ python dBTM.py
```

perform analysis for the outputs.
```{bash}
(envfordbtm)$ python analyze_dBTM.py
```

for OdBTM, just change the command to:
```{bash}
(envfordbtm)$ python setup/poisson_factorization_pretrain_t.py  --data=beauty_makeupalley
(envfordbtm)$ python OdBTM.py
(envfordbtm)$ python analyze_OdBTM.py
```

for BTM:
```{bash}
(envfordbtm)$ python setup/poisson_factorization_individual_t.py  --data=beauty_makeupalley
(envfordbtm)$ python btm.py
(envfordbtm)$ python analyze_BTM.py
```

for TBIP:
```{bash}
(envfordbtm)$ python setup/poisson_factorization_individual_t.py  --data=beauty_makeupalley
(envfordbtm)$ python tbip.py
(envfordbtm)$ python analyze_BTM.py
```

for dJST:
```{bash}
./djst -est -config ../mozilla.train.config
./djst -est -config ../mozilla.test.config
python analyze_dJST.py
```

##Coherence & Uniqueness
Set the local Palmetto with https://github.com/dice-group/Palmetto/wiki/How-Palmetto-can-be-used, then run:
```{bash}
python run6.py
python ave_uniqueness.py
python ave_coherence.py
```

## References
Part of our code is based on: Text-Based Ideal Points by Keyon Vafa, Suresh Naidu, and David Blei (ACL 2020). https://github.com/keyonvafa/tbip 