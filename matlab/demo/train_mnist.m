
clear  
clc  
close all  
format long %设置精度，caffe的损失貌似精度在小数点后面好几位  
addpath('..')  
caffe.reset_all%重设网络，否则载入两个网络会卡住  
solver=caffe.Solver('lenet_solver1.prototxt'); %载入网络  
loss=[];%记录相邻两个loss  
accuracy=[];%记录相邻两个accuracy  
hold on%画图用的  
accuracy_init=0;  
loss_init=0;  
for i=1:10000  
    solver.step(1);%每迭代一次就取一次loss和accuracy  
    iter=solver.iter();  
    loss=solver.net.blobs('loss').get_data();%取训练集的loss  
    accuracy=solver.test_nets.blobs('accuracy').get_data();%取验证集的accuracy  
      
    %画loss折线图  
    x=[i-1,i];  
    y=[loss_init loss];  
    plot(x,y,'r-')  
    drawnow  
    loss_init=loss;  
end  