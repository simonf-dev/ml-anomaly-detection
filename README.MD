# 

# Prepare virtual env

Tutorial for Ubuntu:
This tutorial is mainly for Ubuntu, but it should work on other Unix distributions too, just change 
installation and specific commands for your distribution.
1. We need to download correct version of Python and install pip to it. \
`sudo apt-get install python3.6 && python3.6 -m pip install pip`
   
2. We create virtual environment and activate it. \
` virtualenv ml-python && source ml-python/bin/activate`
   
3. We install pcplog and requirements for the tool. \
`pip3.6 install dist/pcplog-1.0.0-py3-none-any.whl && pip3.6 install -r requirements.in`
   
# Tutorial in general
This tool is for anomaly detection in automatized tests from CQE team in the company Red Hat. 
Tests are executed on the Beaker platform. Then its used pcplog tool from './dist' (which is 
part of the bachelor thesis) to extract all important information from PCP archives and metadata files. All 
information are saved to JSON outputs(all output_NUMBER.json outputs). Then are these JSON 
outputs parsed by number of nodes, types of metrics and types of test_cases to folder './data'. In 
the parsing workflow are solved problems as missing values etc. Datasets are saved by some 
hierarchy in that folder and then are used by ML. There are also pretrained models in './data' 
folder for every test_case,number of nodes etc. Now we have 4 types of functionality in the tool.
All functionality options are executed from command line by starting main.py with commands and 
opts. 
1) We can get output_{ID}.json from pcp_archive and metadata files. (Not working right now, 
   because I need to get data without critical information)
   
2) We can save data from output_{ID}.json to data folder which is used by ML. Data are parsed by 
   number of nodes, test_cases and then saved in some hierarchy defined in settings.py for data 
   folder. If the dataset already exists, its replaced and no duplications arise.
   
3) We can get ML results for concrete output_{ID}.json, models tell us if there are some 
   anomalies or not. We can get output in logical or readable format. 
   
4) We can retrain model for specific test_case, number of nodes, type of data etc. Its taken default
   autoencoder model and data from root folder. Model is then saved in root folder and also 
   threshold is saved for it.
   
I made it like a console application for an easy use and reproducibility, but architecture 
should be done to adapt for REST API etc.
# Fun starts
We have 4 main functions of this tool. Functionality is also combined with installed pcplog tool.
Lets try, what our tool offers. \
`python3.6 main.py --help`

Now we can see basic help to identify all commands. Lets try execute some ML analysis. \
`python3.6 main.py get-result --help`

We have some prepared data in folder exemplary_data. We have trained models for all testcases, 
so now we can check if our exemplary_data are OK or there are anomalies in some testcases. \
`python3.6 main.py get-result -i exemplary_data/output_114711.json -d ./data`

We can see, if there were found some anomalies or not. If we want to logical output, we can just 
use this option. \
`python3.6 main.py get-result -i exemplary_data/output_114711.json -d ./data --logical-output`

Be free to try other options from --help and also another files from 'exemplary_data'. You can 
also use data from the folder 'pcp_raw_data', but models are trained from them and results 
wouldn't tell too much.

Now we can train another function. Imagine situation that we made ML analysis on some output as 
above. Now we know, that output is correct and we want to save him to next retrain of the model. \
`python3.6 main.py save-result -o ./data -i ./exemplary_data/output_114711.json`

What if only one of the testcase ended correct and we only want to save that concrete testcase? \
`python3.6 main.py save-result -l testcase_3 -o ./data -i ./exemplary_data/output_114711.json`

Be free to use this command more times, because every output has unique id and we avoid 
duplications by changing row in the ML data.

So now we saved some new data and we want to retrain the model. Retraining of the model could be 
really hard for your CPU or GPU and also could be very long. So be free to skip this step.
We will only retrain model for testcase_3 and only for cpus metrics. \
`python3.6 main.py retrain-model -l testcase_3 -d ./data --skip-memory`


TODO: Last functionality of the tool is to generate output JSON from pcp_archive and metadata 
files(that JSON files, that are used in other functionality options), we need to generate 
archives on nonlicensed 
distributions,
so 
functionality is 
working ok,
but there are no available data yet.