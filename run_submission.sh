# chmod +x run_submission.sh
pip3 install -e .
python3 genjson.py
rm -rf ../primitives/v2019.6.7/ICSI
cp -r ICSI ../primitives/v2019.6.7/
git --git-dir=../primitives/.git --work-tree=../primitives add -u
git --git-dir=../primitives/.git --work-tree=../primitives add v2019.6.7/ICSI/*
git --git-dir=../primitives/.git --work-tree=../primitives commit -m "updating ICSI primitives and pipelines"
git --git-dir=../primitives/.git --work-tree=../primitives push
