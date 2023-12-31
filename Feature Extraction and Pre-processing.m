%Plotting inter-stride time intervals and using the midcross function to plot the location of each crossing of the right foot only
%First 10 crossings were excluded to remove transients. Tolerance was set to 25%

control1m=load('control1m.mat');
time=linspace(0,60,length(hunt14m.val(1,:)));

figure;
plot(time,hunt14m.val(1,:));
hold on 
plot(time,hunt14m.val(2,:));
legend('Left foot','Right foot');
xlabel('Time (sec)');
ylabel('mV');
grid on
title('hunt14m');
filename = 'figure_1.jpg';
print(filename, '-djpeg');

Fs = 300;
gaitSignal =hunt14m.val;
figure
midcross(gaitSignal(1,:),Fs,'tolerance',25);
xlim([0 30])
xlabel('Sample Number')
ylabel('mV')
title('hunt14m');
filename = 'figure_2.jpg';
print(filename, '-djpeg');

figure;
plot(time,control1m.val(1,:));
hold on 
plot(time,control1m.val(2,:));
legend('Left foot','Right foot');
xlabel('Time (sec)');
ylabel('mV');
grid on
title('control1m');
filename = 'figure_3.jpg';
print(filename, '-djpeg');

Fs = 300;
gaitSignal =control1m.val;
figure
midcross(gaitSignal(1,:),Fs,'tolerance',25);
xlim([0 30])
xlabel('Sample Number')
ylabel('mV')
title('control1m');
filename = 'figure_4.jpg';
print(filename, '-djpeg');

%Plot graph of against time between strides(sec) against stride number for the right foot of patients labelled ‘control 1-5’, 'hunt4’,‘hunt13’, ‘hunt1’, ‘hunt8’ and ‘hunt5’(10 records).Exclude the first 10 crossings to remove transients.
control1m=load('control1m.mat');
control2m=load('control2m.mat');
control3m=load('control3m.mat');
control4m=load('control4m.mat');
control5m=load('control5m.mat');
hunt1m=load('hunt1m.mat');
hunt4m=load('hunt4m.mat');
hunt5m=load('hunt5m.mat');
hunt8m=load('hunt8m.mat');
hunt13m=load('hunt13m.mat');

Fs=300;
pnames = {"control1m.val","control2m.val","control3m.val","control4m.val",...
    "control5m.val","hunt1m.val","hunt4m.val","hunt5m.val","hunt8m.val",...
    "hunt13m.val",};
for i = 1:10
  gaitSignal = eval(pnames{i});
  IND2 = midcross(gaitSignal(1,:),Fs,'Tolerance',25);
  IST{i} = diff(IND2(9:2:end));   
  varIST(i) = var(IST{i});
end
figure
hold on
for i = 1:5
  plot(1:length(IST{i}),IST{i},'.-r')
  plot(1:length(IST{i+5}),IST{i+5},'.-b')
end
xlabel('Stride Number')
ylabel('Time Between Strides (sec)')
legend('Hunt','Control')
grid on
filename = 'figure_5.jpg';
print(filename, '-djpeg');

%Aligning and comparing signals from the left and right foot records using dynamic time warping for ‘control1’ and ‘hunt14’as an example
%Calculating the Euclidean distance

hunt14m=load('hunt14m.mat');
control1m=load('control1m.mat');

x1=hunt14m.val(1,:);
y1=hunt14m.val(2,:);

figure
dtw(x1,y1);
legend('Left foot','Right foot');
filename = 'figure_6.jpg';
print(filename, '-djpeg');

x2=control1m.val(1,:);
y2=control1m.val(2,:);

figure
dtw(x2,y2);
legend('Left foot','Right foot');
filename = 'figure_7.jpg';
print(filename, '-djpeg');

%Construct Feature Vector
pnames = {"control1m.val","control2m.val","control3m.val","control4m.val",...
    "control5m.val","hunt1m.val","hunt4m.val","hunt5m.val","hunt8m.val",...
    "hunt13m.val",};
for i = 1:10
  gaitSignal = eval(pnames{i});
  IND2 = midcross(gaitSignal(1,:),Fs,'Tolerance',25);
  IST{i} = diff(IND2(9:2:end));   
  varIST(i) = var(IST{i});
  feature2(i) = dtw(gaitSignal(1,:),gaitSignal(2,:));
end

feature1 = varIST;

figure
plot(feature1(1:5),feature2(1:5),'r*',...
    feature1(6:10),feature2(6:10),'b+',...
    'MarkerSize',10,'LineWidth',1)
xlabel('Variance of Inter-Stride Times')
ylabel('Distance Between Segments')
legend('Hunt','Control')
grid on
filename = 'figure_8.jpg';
print(filename, '-djpeg');

Reference: https://www.mathworks.com/help/signal/ug/extracting-classification-features-from-physiological-signals.html
