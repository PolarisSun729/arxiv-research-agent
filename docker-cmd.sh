# =============== 创建RAG网络 ===============

docker network create ragnet --subnet=192.180.0.0/16

# =============== Python 容器 ===============

docker run --network ragnet --name rag-python -it \
  -p 8000:80 \
  -v /Users/yakoo5/Documents/dev/github/huangjia2019:/mnt/huangjia2019 \
  anolis-registry.cn-zhangjiakou.cr.aliyuncs.com/openanolis/python:3.11.1-8.6 \
  /bin/bash

# =============== Node 容器 ===============

docker run --network ragnet --name rag-node -it \
  -p 8001:80 \
  --volumes-from rag-python \
  anolis-registry.cn-zhangjiakou.cr.aliyuncs.com/openanolis/node:16.17.1-nslt-8.6 \
  /bin/bash

# =============== 删除 容器、网络 ===============

# docker rm -f rag-python
# docker rm -f rag-node
# docker network rm ragnet