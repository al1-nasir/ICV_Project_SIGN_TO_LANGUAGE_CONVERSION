import pandas as pd

from data_processing import direct_csv_to_video, video_files
from data_processing import filter_and_save_brady_rows
from data_processing import play_video
from data_processing import get_all_files_from_directory

# directory path in WSL format
main_dir = "/mnt/c/Users/Ali Nasir/ICV_batches"

video_files = get_all_files_from_directory(main_dir)


# adding asllvd_signs path
sign_csv_path = './dataset/asllvd_signs.csv'
sign_csv = pd.read_csv(sign_csv_path)

# getting just csv with Brady name
# Brady_signs = filter_and_save_brady_rows(sign_csv_path,'Brady_ASL_Signs.csv',column_name='full video file'
#                                          ) # it has been done

Brady_Signs = pd.read_csv('./dataset/Brady_ASL_Signs.csv')

index = 15
Brady_videos_path = direct_csv_to_video(Brady_Signs['full video file'][index] ,
                    str(Brady_Signs['start frame of the sign (relative to full videos)'][index]) ,
                    str(Brady_Signs['end frame of the sign (relative to full videos)'][index]))



print(Brady_videos_path)
# print(Brady_Signs.iloc[index])
# play_video(Brady_videos_path)


