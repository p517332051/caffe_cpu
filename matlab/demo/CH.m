function CH
clear  
clc  
close all  
format long
caffe.reset_all;
caffe_log_file_base = fullfile('CH_log','caffe_log_CH');
caffemodel = 'E:\caffe-rfcn-r-fcn1\matlab\demo\example_IQA2D_iter_5000(1).caffemodel';
caffe.init_log(caffe_log_file_base);
caffe_solver=caffe.Solver('E:\caffe-rfcn-r-fcn1\matlab\demo\solver12.prototxt');
caffe_solver.net.copy_from(caffemodel);

%%
im_data = imread('27.bmp');
im_data = prepare(im_data);
lables = 1;
net_input = {im_data,lables};
caffe_solver.net.set_input_data(net_input);
max_iter = 1000;
loss = [];
iter_ = caffe_solver.iter()
while iter_ < max_iter
    caffe_solver.net.set_phase('train');    
    caffe_solver.step(1);
    rst = caffe_solver.net.get_output();  
    loss = [loss rst.data];     
    iter_ = caffe_solver.iter()
%     x=[iter_-1,iter_];  
%     plot(x,loss,'r-')
%     drawnow 
end
plot(loss,'r-')
% lables_ = caffe_solver.net.blobs('labels').get_data();
% rst = caffe_solver.net.get_output();
end
function im_data = prepare(im_data)
        %%
% 由matlab输入到caffe中图片要做的处理
    %im_data = imread('./examples/images/cat.jpg'); % read image
    im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % permute width and height
    im_data = single(im_data); % convert to single precision

end