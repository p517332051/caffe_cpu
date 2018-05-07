
clear  
clc  
close all  
format long %���þ��ȣ�caffe����ʧò�ƾ�����С�������ü�λ  
addpath('..')  
caffe.reset_all%�������磬����������������Ῠס  
solver=caffe.Solver('lenet_solver1.prototxt'); %��������  
loss=[];%��¼��������loss  
accuracy=[];%��¼��������accuracy  
hold on%��ͼ�õ�  
accuracy_init=0;  
loss_init=0;  
for i=1:10000  
    solver.step(1);%ÿ����һ�ξ�ȡһ��loss��accuracy  
    iter=solver.iter();  
    loss=solver.net.blobs('loss').get_data();%ȡѵ������loss  
    accuracy=solver.test_nets.blobs('accuracy').get_data();%ȡ��֤����accuracy  
      
    %��loss����ͼ  
    x=[i-1,i];  
    y=[loss_init loss];  
    plot(x,y,'r-')  
    drawnow  
    loss_init=loss;  
end  