import cv2
import os

# get the names of folders
path_folders  = '../../../DataSet/profession_dataset/'
folders = os.listdir(path_folders)
folders.sort()

for folder in folders:
    path_in = path_folders + folder + '/rally_video/'
    
    # get the filenames of the mp4 file
    filenames = os.listdir(path_in)

    for filename in filenames:
        if ('.mp4' in filename) and ('predict' not in filename):
            video = filename.split('.mp4')[0]
            path_out = '../../../DataSet/profession_dataset/%s/frame/%s/'%(folder, video)
            fn_video = path_in + video + '.mp4'
            os.makedirs(path_out, exist_ok=True) # create the folder if not exist
            
            # extract frames from mp4
            vidcap = cv2.VideoCapture(fn_video)
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite("%s%d.png"%(path_out, count), image) # save frame as png file
                success,image = vidcap.read()
                count += 1
            print('Create %d frames for %s.mp4 in %s folder.'%(count, video, folder))
    
    
    
