((nohup python3 TrainCore320_Inception_LAO.py >train_angle_lao.out 2>&1) ; $(nohup python3 TrainCore320_Inception_RAO.py >train_angle_rao.out 2>&1) ; $(nohup python3 TrainCore320_Inception_CRA.py >train_angle_cra.out 2>&1))&


d:
cd D:\OneDrive\PycharmProjects\InceptionV3
python TrainCore320_Inception_CRA.py >> E:\train_angle_cra.out
python TrainCore320_Inception_RAO.py >> E:\train_angle_rao.out

