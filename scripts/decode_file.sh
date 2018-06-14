
#Comparing wave and mspec
#Gain normalisation is crucial
python3 ./src/offline_ser.py -p_mode 2 -f_mode 1 --wav ./wav/NEWS.16k.wav -log ./output/news16k.file.norm.wav.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --save -g_min -0.284261 -g_max 0.317006
python3 ./src/offline_ser.py -p_mode 2 -f_mode 0 --wav ./wav/NEWS.16k.wav -log ./output/news16k.file.norm.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.1.h5 -c_len 10 -m_t_step 100 -tasks 'arousal:3,valence:3' --save -g_min -0.284261 -g_max 0.317006 -three_d

#MORE NEWS
#Check gain range first
python ~/Workspace/Workspace_surf/Exp/feature_ext/scripts/extract_feat_temporal_LLD_rosa.py -f ~/Workspace/Workspace_old_tf_keras/LIVE_SER/wav/ -m ~/Workspace/Workspace_old_tf_keras/LIVE_SER/wav/meta.CNNNEWS.txt --gain_stat

python ./src/offline_ser.py -p_mode 2 -f_mode 1 -wav ./wav/2017-10-02_2300_US_CNN_Erin_Burnett_OutFront.16k.wav -log ./output/2017-10-02_2300_US_CNN_Erin_Burnett_OutFront.16k.wav.class.raw.live.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' -g_min -0.648979 -g_max 0.54826



#Comparing wave, mspec and their frame and file mode
#They are all similar to each other.
python ./src/offline_ser.py -p_mode 1 -f_mode 1 --wav './wav/IEMOCAP_EXCITED.16k.wav' -log ./output/iemocap.file.norm.wav.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --save -g_min -0.284261 -g_max 0.317006
python ./src/offline_ser.py -p_mode 1 -f_mode 0 --wav './wav/IEMOCAP_EXCITED.16k.wav' -log ./output/iemocap.file.norm.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.1.h5 -c_len 10 -m_t_step 100 -tasks 'arousal:3,valence:3' --save -g_min -0.284261 -g_max 0.317006 --three_d
python ./src/offline_ser.py -p_mode 1 -f_mode 1 --wav './wav/IEMOCAP_EXCITED.16k.wav' -log ./output/iemocap.frame.norm.wav.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' -g_min -0.284261 -g_max 0.317006
python ./src/offline_ser.py -p_mode 1 -f_mode 0 --wav './wav/IEMOCAP_EXCITED.16k.wav' -log ./output/iemocap.frame.norm.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.1.h5 -c_len 10 -m_t_step 100 -tasks 'arousal:3,valence:3' -g_min -0.284261 -g_max 0.317006 --three_d

#live demo with a new keras trained model (2.xxx)

#For linux os, you need to set a default device for pulseaudio
pacmd list-sources | grep -e device.string -e 'name:'
#For example, you find "name: <Channel_1__Channel_2.4>" as your input source.
pacmd "set-default-source Channel_1__Channel_2.4"
#Then, the following script will use the default pulse device
python ./src/offline_ser.py -p_mode 2 -f_mode 1 -log ./output/live.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'

#For mac os, you need to set a device for "offline_ser.py"
#Running the script without arguments, it will show you a list of devices.
python ./src/offline_ser.py

#If you find your device in index 2
#The following script uses automatic gain control
python ./src/offline_ser.py -d_id 1 -p_mode 1 -f_mode 1 -log ./output/live.wav.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --seq2seq
python ./src/offline_ser.py -d_id 1 -p_mode 1 -f_mode 0 -log ./output/live.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.1.h5 -c_len 10 -m_t_step 100 -tasks 'arousal:3,valence:3' --three_d --seq2seq
python ./src/offline_ser.py -d_id 1 -p_mode 1 -f_mode 0 -log ./output/live.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.1.4cls.h5 -c_len 10 -m_t_step 100 -tasks 'nhsa:4' --three_d --stl --seq2seq

python ./src/offline_ser.py -d_id 1 -p_mode 1 -f_mode 0 -log ./output/live.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.gar.h5 -c_len 10 -m_t_step 100 -tasks 'gender:2,acted:2,arousal:3' --three_d --seq2seq

python ./src/offline_ser.py -p_mode 1 -f_mode 0 --wav './wav/IEMOCAP_EXCITED.16k.wav' -log ./output/iemocap.file.norm.mspec.csv -md ./model/si.ENG.cw.mspec_mm.3d.rc3d.gar.h5 -c_len 10 -m_t_step 100 -tasks 'gender:2,acted:2,arousal:3' -g_min -0.284261 -g_max 0.317006 --three_d --seq2seq

#Live demo for laughter detection
python ./src/offline_ser.py -d_id 1 -vd 1000 -p_mode 2 -f_mode 0 -log ./output/live.mspec.csv -md ../SER_KERAS_TF_TRAINER/model/ami.laugh.mspec.cnnlstm.0.h5 -c_len 10 -m_t_step 100 -tasks 'laughter:2' --stl --save
python ./src/offline_ser.py -d_id 1 -vd 1000 -p_mode 2 -f_mode 1 -log ./output/live.wav.csv -md ../SER_KERAS_TF_TRAINER/model/ami.raw.cnnlstmfcn.0.h5 -c_len 1600 -m_t_step 16000 -tasks 'laughter:2' --stl --save

