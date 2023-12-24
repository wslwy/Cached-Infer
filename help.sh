# activate env
# 404
conda activate lwy-py39-neuro
# 407
conda activate wyliang-py39 

# 设置 python Path 环境变量
export PYTHONPATH=/home/wyliang/Neurosurgeon:$PYTHONPATH

# cloud 
python cloud_api.py -i 127.0.0.1 -p 9999 -d cpu

# edge
python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net

# push to remote repo
git push -v origin main

# server
python server_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net

# client
python client_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net

