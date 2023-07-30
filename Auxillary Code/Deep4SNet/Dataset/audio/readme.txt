
% This code allows recovering the original voice recording from the fake voice recording
% The inputs are:
%		fa is the fake voice recording
%		key is the secret key.
% The output is:
% 		roriginal is the recovered original voice recording


[fa, FS]=audioread('name1'); %name1: name of the fake voice recording. Example: 'fake1_1.wav'

fileID = fopen('name2','r'); %name2: name of the txt file. Example: 'key_fake1_1.txt'
formatSpec = '%d';
key = fscanf(fileID,formatSpec);

[C,L] = wavedec(fa,4,'db10');

Cm(key)=C; 
roriginal = waverec(Cm,L,'db10'); % Recovered original voice signal

sound(roriginal, FS);



