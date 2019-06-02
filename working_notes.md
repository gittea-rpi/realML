# Directory structure
I store everything on Dropbox, because if I accidentally delete files (happens a lot with me when I'm using git), I can easily restore them. I use the following directory stucture:

- Base directory for everything : `~/Dropbox/research/realML`
- the realML repo clone:`~/Dropbox/research/realML/realML`
- Base directory for repos from the d3m servers: `~/Dropbox/research/realML/d3mrepos`
- subdirectories of the cloned d3m repos: 
	- `ta1-submission-pipelines`: my fork of the ta1 pipeline submission repo that I push my pipelines to
	
# Setting up a Docker d3m environment to work locally
This lets you make changes to your primitives and try them out in pipelines using the d3m python images on your machine locally, so you can do more effective debugging

## Setting up the docker environment
Read through the following instructions before starting to follow them.

1. In the following, I left out some startup steps that I did last year: docker log into the d3m registry and download the latest D3M image, for example. ***Ben, please add this if you do it, just for more complete documentation.***

2. Set some useful environment variables on the host system. One is the docker id for the d3m image, one is a directory that will be mounted onto the docker container so we can transfer files to and from the host system, and one is the key file for sshing into Github (where the RealML code repo is) and sshing into the D3M servers.

   ```
   export D3MIMAGE=5c314b00a752
   export DATAIN=~/Dropbox/research/RealML/pipelines
   export D3MKEY=~/.ssh/gitlab_rsa # this is also my github key, so no need for a separate var
   ```

3. Now copy the data you want to be accessible from the host to the Docker container into the mounted directory, and your ssh key(s).

4. Launch a Docker container using the latest D3M image and set up some useful variables inside the container.
   
   ```
   docker run -it -v $DATAIN:/root/datain $D3MIMAGE /bin/bash

   export REALMLREPO=https://github.com/ICSI-RealML/realML.git
   export D3MPRIMITIVEREPO=git@gitlab.com:alexgittens/primitives.git
   export D3MUPSTREAMREPO=https://gitlab.com/datadrivendiscovery/primitives
   export GITUNAME="Alex Gittens"
   export GITEMAIL="gittea@rpi.edu"
   export D3MPRIVATEKEY="gitlab_rsa"
   export D3MPUBLICKEY="gitlab_rsa.pub"
   export LOCALREALML=~/realML
   export LOCALPRIMITIVES=~/primitives_repo/v2019.4.4
    ```

5. Setup your ssh keys and and git user information
    
    ```
    mkdir ~/.ssh
    chmod u=rwx,g= ~/.ssh
    cp ~/datain/$D3MPRIVATEKEY ~/.ssh/
    cp ~/datain/$D3MPUBLICKEY ~/.ssh/
    chmod u=rw,go= ~/.ssh/$D3MPRIVATEKEY
    chmod u=rw,go=r ~/.ssh/$D3MPUBLICKEY
    git config --global user.name $GITUNAME
    git config --global user.email $GITEMAIL
    ```

6. Setup ssh so that it uses the correct key file
    
    ```
    cat << ENDCONFIG > ~/.ssh/config
    Host gitlab.com
    	HostName gitlab.com
    	IdentityFile ~/.ssh/$D3MPRIVATEKEY
    
    Host gitlab.datadrivendiscovery.org
    	IdentityFile ~/.ssh/$D3MPRIVATEKEY
    	
    Host github.com
    	HostName github.com
    	IdentityFile ~/.ssh/$D3MPRIVATEKEY
    ENDCONFIG
    ```

7. Clone the fork of the d3m primitive repo and CHECK OUT THE UPDATE BRANCH (don't work in master!); Also, remember to occasionally pull from the original JPL repo to update the master branch. Of course, here I already have a branch icsiupdate I'm using to work in, you may have to create one.
 	
 	```
 	git clone $D3MPRIMITIVEREPO
 	cd primitives
	git remote add upstream $D3MUPSTREAMREPO
	git fetch upstream
	git checkout master
	git merge upstream/master
	git push

 	git checkout --track origin/icsiupdate
 	```
 	
8. Clone the realML repo and install it as a working package: this is necessary so that the primitive json annotations link to the correct git commits of the realML primitive repo. Everytime you do a commit then regenerate the
json annotations, they will point to a different git commit.
	
	```
	git clone $REALMLREPO
	cd realML
	pip3 install -e .
	```
	
9. Do your work and modifications on the primitives and pipelines, using the fact that you can run the d3m primitives and so on inside the container.

10. ***COMMIT AND PUSH AFTER EVERY CHANGE*** so that when you generate the json annotations they point to the right commit.

11. Go back to the realML/realML directory, and generate the json annotations in the ICSI subdirectory (this deletes old content of that directory, if it exists)

    ```
    python3 genjsons.py
    ```

12. Copy the annotations to our fork of the D3M primitive repo and push them to automatically start the validation procedure running. Once it has run successfully, you can submit a merge request.

    ```
    rm -rf $LOCALPRIMITIVES/ICSI
    cp -r $LOCALREALML/ICSI $LOCALPRIMITIVES
    git add $LOCALPRIMITIVES/*
    git commit -m "updating ICSI primitives and pipelines" 
    git push
    ```
    
13. Now, you need to get the pipelines onto your host system (through the mounted directory, presumably), then *on the host system* copy the pipelines to the ta1-submission repo, replace the old pipelines and push, wait for gitlab to rebuild the pipeline submission image through CI (or force it to the gitlab console), and run the pipelines on the jump server

    ```
    cd ~/Dropbox/research/RealML/d3mrepos/ta1-submission-pipelines
    rm -rf pipelines/*
    cp ../../realML/ICSI/*/*/pipelines/*.json pipelines #change to reflect locations for your pipelines
    git commit -a -m "updated pipelines"
    git push
    	
    ssh ta1-icsi@k8s.datadrivendiscovery.org -i ~/.ssh/ICSI_D3M_id_rsa
    cd ta1-example
    /performer-toolbox/d3m_runner/d3m_runner.py --mode ta1 --yaml-file ./generateSubmission.yml --debug
    ```
