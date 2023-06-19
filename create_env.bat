call deactivate
call conda remove -n dis --all -y
call conda create -n dis python=3.7  -y
call conda activate dis

call pip install gradio  pillow  chardet -i https://pypi.tuna.tsinghua.edu.cn/simple --user
call conda install  scikit-image scipy pandas matplotlib opencv pyyaml tqdm  pywavelets -y
@REM call conda install opencv -y
call pip install --force-reinstall charset-normalizer==3.1.0
call conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch -y
