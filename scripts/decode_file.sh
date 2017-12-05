#check g_min g_max
python ~/Workspace/Workspace_surf/Exp/feature_ext/scripts/extract_feat_temporal_LLD_rosa.py -f ~/Workspace/Workspace_old_tf_keras/squirrel_ser/wav/ -m ~/Workspace/Workspace_old_tf_keras/squirrel_ser/wav/meta.IEMOCAP.txt --gain_stat

python ~/Workspace/Workspace_surf/Exp/feature_ext/scripts/extract_feat_temporal_LLD_rosa.py -f ~/Workspace/Workspace_old_tf_keras/squirrel_ser/wav/ -m ~/Workspace/Workspace_old_tf_keras/squirrel_ser/wav/meta.news.txt --gain_stat

#decoding without VAD
#python ./src/decoding.py -p_mode 2 -f_mode 0 -l_wav ./wav/IEMOCAP_EXCITED.16k.wav -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006

#python ./src/decoding.py -p_mode 2 -f_mode 1 -l_wav ./wav/IEMOCAP_EXCITED.16k.wav -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks arousal:3,valence:3 -g_min -0.284261 -g_max 0.317006

#python ./src/decoding.py -p_mode 2 -f_mode 1 -l_wav ./wav/NEWS.16k.wav -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks arousal:3,valence:3 -g_min -0.564745 -g_max 0.55982

#VAD mode
#MSPC 2D-CNN-LSTM
#file mode
#python ./src/offline_ser.py -p_mode 2 -f_mode 0 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.dist.mfcc.file.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006 --save

#live mode
#python ./src/offline_ser.py -p_mode 2 -f_mode 0 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.dist.mfcc.live.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006

#classification mode
python ./src/offline_ser.py -p_mode 1 -f_mode 0 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.class.mfcc.file.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006 --save

python ./src/offline_ser.py -p_mode 1 -f_mode 0 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.class.mfcc.live.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006

#RAW 1D-CHRM
python ./src/offline_ser.py -p_mode 1 -f_mode 1 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.class.raw.live.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' 

python ./src/offline_ser.py -p_mode 1 -f_mode 1 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.class.raw.file.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --save

python ./src/offline_ser.py -p_mode 1 -f_mode 1 -wav ./wav/IEMOCAP_SAD.16k.wav -log ./wav/IEMOCAP_SAD.16k.wav.class.raw.file.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --save

#python ./src/offline_ser.py -p_mode 2 -f_mode 1 -wav ./wav/IEMOCAP_EXCITED.16k.wav -log ./wav/IEMOCAP_EXCITED.16k.wav.dist.raw.live.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'

#NEWS
python ./src/offline_ser.py -p_mode 1 -f_mode 1 -wav ./wav/NEWS.16k.wav -log ./wav/NEWS.16k.wav.class.raw.live.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'

python ./src/offline_ser.py -p_mode 1 -f_mode 0 -wav ./wav/NEWS.16k.wav -log ./wav/NEWS.16k.wav.class.mfcc.live.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006

python ./src/offline_ser.py -p_mode 1 -f_mode 1 -wav ./wav/NEWS.16k.wav -log ./wav/NEWS.16k.wav.class.raw.file.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3' --save

python ./src/offline_ser.py -p_mode 1 -f_mode 0 -wav ./wav/NEWS.16k.wav -log ./wav/NEWS.16k.wav.class.mfcc.file.csv -md ./model/MSPEC_MM.all.ar_vl.0.h5 -c_len 10 -m_t_step 500 -tasks arousal:2,valence:2 -g_min -0.284261 -g_max 0.317006 --save