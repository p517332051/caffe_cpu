function CH
clear  
clc  
close all  
format long
caffe.reset_all;
caffe_log_file_base = fullfile('CH_log','caffe_log_CH');
caffe.init_log(caffe_log_file_base);
net=caffe.Net('ResNet-50-deploy.prototxt','test');

%%
% im_data = imread('27.bmp');
im_data = zeros(600,1000,3);
im_data = prepare(im_data);
net_input = {im_data};
net.reshape_as_input(net_input);
net.forward(net_input);
size(net.blobs('res4f').get_data())
size(net.blobs('res5c').get_data())


end
function im_data = prepare(im_data)
        %%
% 由matlab输入到caffe中图片要做的处理
    %im_data = imread('./examples/images/cat.jpg'); % read image
    im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % permute width and height
    im_data = single(im_data); % convert to single precision

end