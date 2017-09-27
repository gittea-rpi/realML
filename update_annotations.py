#!/usr/bin/env python3

"""
update_annotations.py basedir newversion

assume basedir contains the primitives as subdirectories. it goes them and updates their versions to newversion, and computes the appropriate id. it also renames
the version directories to newversion

e.g.
update_annotations.py primitive_annotations 0.1.2
"""

import sys
import json
import os
import shutil
from uuid import uuid3, NAMESPACE_DNS
from collections import OrderedDict

basedir = sys.argv[1]
newversion = sys.argv[2]

def getid(primitive, vers):
    return str(uuid3(uuid3(NAMESPACE_DNS, 'datadrivendiscovery.org'), primitive + str(vers)))

for root, dirs, files in os.walk(basedir, topdown=False):
    for name in files:
        if name.endswith(".json"):
            with open(os.path.join(root, name), 'r') as fin:
                injson = json.load(fin, object_pairs_hook=OrderedDict)
            injson['version'] = newversion
            injson['id'] = getid(injson['name'], newversion)
            with open(os.path.join(root, name), 'w') as fout:
                json.dump(injson, fout, indent=2) 
    for name in dirs:
        #assumes that if the dir contains a primitive.json, it's a version so needs to be renamed
        if os.path.isfile(os.path.join(root, name, "primitive.json")):
            shutil.move(os.path.join(root, name), os.path.join(root, newversion))

