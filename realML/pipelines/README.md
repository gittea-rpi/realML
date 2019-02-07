- each pipeline file generates a json for a pipeline for the corresponding primitive when run as a script
- `runpipelinejson.py' takes a json file as input, then runs it on a specific (regression) test dataset
- `explicitpipeline.py` contains a programmatic construction of a pipeline (for the FastLAD primitive), which can be run in the python shell step by step to check the output of each stage and inspect metadata, etc.

