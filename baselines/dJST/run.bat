#!/bin/bash
# training the DJST model
src/djst -est -mode sliding -nscales 3 -nsentiLabs 3 -ntopics 5 -niters 1000 -savestep 200 -updateParaStep 40 -twords 20 -data_dir data/train -result_dir out/train -datasetFile train_epoch -sentiFile data/mpqa.constraint


# inference on the new data using the trained model
src/djst -inf -model epoch_0-final -niters 40 -savestep 20 -model_dir out/train -data_dir data/test -result_dir out/test -datasetFile test.dat -sentiFile data/mpqa.constraint

