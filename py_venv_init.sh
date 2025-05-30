pip3 install -r ../../requirements_mac.txt
pip3 install -r ../../requirements_mac.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 --default-timeout=1000 install -r ../../requirements_mac.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip3 --default-timeout=1000 install -r ../../requirements_mac_py_3.11.9.txt -i https://mirrors.aliyun.com/pypi/simple

pip3 --default-timeout=1000 install onnxruntime==1.20.1 -i https://mirrors.aliyun.com/pypi/simple

python3 -V
Python 3.13.3

------------------------ 默认镜像 ------------------------

python3.11 -m venv venv_default

source venv_default/bin/activate

pip3.11 --default-timeout=1000 install -r requirements_mac.txt

# ------------------------ [MacOS Sonama 14.5 (Apple M1 芯片)] Homebrew 更换为 清华 镜像源 ------------------------

# https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/
export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"
# https://mirrors.tuna.tsinghua.edu.cn/help/homebrew-bottles/
export HOMEBREW_API_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles"

# ------------------------ [MacOS Sonama 14.5 (Apple M1 芯片)] 安装 Python3.11 ------------------------

# 本机默认安装的 python3 版本过低，不满足要求
python3 -V
# ==> Python 3.9.6

# 搜索支持安装的 python3 版本
brew search python@
# ==> Formulae
# python@3.10  python@3.12  python@3.8   bpython      wxpython     cython       ptpython
# python@3.11  python@3.13  python@3.9   ipython      pythran      jython

brew install python@3.11

# Python is installed as
#   /usr/local/bin/python3.11
# You can install Python packages with
#   pip3.11 install <package>

python3.11 -V
# ==> Python 3.11.12

# ------------------------ 清华镜像 ------------------------

python3.11 -m venv venv_tsinghua

source venv_tsinghua/bin/activate

pip3.11 --default-timeout=1000 install -r requirements_mac_py_3.11.9.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ------------------------ Aliyun 镜像 ------------------------

python3.11 -m venv venv_aliyun

source venv_aliyun/bin/activate

pip3.11 --default-timeout=1000 install -r requirements_mac_py_3.11.9.txt -i https://mirrors.aliyun.com/pypi/simple