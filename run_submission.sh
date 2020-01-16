# chmod +x run_submission.sh
pip3 install -e .
python3 genjson.py
rm -rf ../primitives/v2020.1.9/ICSI
cp -r ICSI ../primitives/v2020.1.9/
git --git-dir=../primitives/.git --work-tree=../primitives add -u
git --git-dir=../primitives/.git --work-tree=../primitives add v2020.1.9/ICSI/*
git --git-dir=../primitives/.git --work-tree=../primitives commit -m "updating ICSI primitives and pipelines"
git --git-dir=../primitives/.git --work-tree=../primitives push
