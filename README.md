# shot_boudary_detector
Shot Boundary Detector based on convolutional neural network created by Gygli --> https://arxiv.org/pdf/1705.08214.pdf

To use the shot boundary detector the following python libraries are reuqired
1. moviepy
2. pillow (PIL)
3. pytorch
4. cuda 

For easy use of the shot boundary detector place the video file in the same directory is possible. 
Clone repo and to detect the frames of the input video that are potentially part of a shot boundary run the following command while inside the repo:

For gpu users  --> python get_transition_frames_gpu.py <video_file_path> <name of text file to save predictions>
For cpu users --> python get_transition_frames_cpu.py <video_file_path> <name of text file to save predictions>

The file name to svae files to must have the '.txt' extension added to it. The resulting text file located in 'predictions/<your_text_file_name>' will have the frame numbers in the input video that are part of one of the following transitions:
1. Hard cut
2. Crop cut
3. Dissolve
4. Fade In
5. Fade Out
6. Wipe 

The model used obtains average recall, precision and f1 score of 0.743, 0.741 and 0.742 this is tested on the RAI benchmark

If the requirement is not just to use the shot boundary detector for inference but to improve the operation via traininga and test for new results the repo https://github.com/oladeha2/gygli_sbd_train_and_test can be visited 

This repo contains python scripts required for local dataset creation, training and testing on the RAI videos whose frame content are already available in the repo
