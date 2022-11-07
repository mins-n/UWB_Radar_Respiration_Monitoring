% biopac과 비교용
% fastindex 당 0.6445cm slowindex 1초 당 17index

Windowsize = 510; % 30s 
Window_sliding = 1; % 윈도우 어디부터 시작할지 ex) 1이면 0 ~ 510
Window_rawdata = [];
fast_to_m = 0.006445; % fast index to meter
UWB_Radar_index_start = 0.22; % UWB radar range 0.22 ~ 2.22m
UWB_Radar_index_start = floor(UWB_Radar_index_start/fast_to_m);

%% Baseline threshold
SD = [];
Window_rawdata = rawdata( : , Windowsize*(Window_sliding-1)+1:Windowsize*Window_sliding);

SD = std(rawdata, 0, 2); %거리에 대한 표준편차 배열

[Max, Index] = max(SD); % Max : 가장 큰 표준편차 값, Index : Max의 위치 값 앞쪽 50cm 이내는 제거

Pm = mean(SD(Index-1 : Index+1)); % 가장 큰 표준편차와 -1 index +1 index 평균값

Index = Index + UWB_Radar_index_start;

d0 = Index*fast_to_m; % 가장 높은 편차의 fastindex
n = mean(SD); % 편차 배열의 평균

Baseline_threshold = (Pm - n)/(2*d0 + 1) + n;

%% Dynamic threshold
di = [];
k = [];

di = 1 : size(rawdata,1);
di = di + UWB_Radar_index_start;
di = di .*fast_to_m;
k = di.^2 ./ d0^2;
k = k.^-1;
Dynamic_threshold = k'.*Baseline_threshold;

hold on
plot(Dynamic_threshold( : ));
plot(SD(:));
hold off

%% Threshold crossing
TC_matrix = [];
Data = [];
Distance = [];

TC_matrix = SD > Dynamic_threshold; %SD가 threshold보다 큰 신호는 1, 아니면 0인 행렬
TC = TC_matrix; % TC_matrix 값 확인용
TC_cnt = 0; %threshold를 넘는 연속된 점 수
Human_cnt = 0;

for i = 1 : size(rawdata,1) % 행 길이 만큼 반복
    if (TC_matrix(i))
        TC_cnt = TC_cnt + 1; %점 수 세기
    else
        if(TC_cnt > 0)
            if(TC_cnt < 15)  % 15미만의 점은 특이치로 판별하여 버림
                TC_matrix(i - TC_cnt : i - 1, :) = 0;
                TC_cnt = 0;
            elseif(TC_cnt > 45) % 45이상의 점은 특이치로 판별하여 버림
                TC_matrix(i - TC_cnt : i - 1, :) = 0;
                TC_cnt = 0;
            else % 15이상 45이하 사람으로 판별
                Human_cnt = Human_cnt + 1;  %사람의 수 카운트
                Distance(Human_cnt, 2) = 0;
                Distance(Human_cnt, :) = [i - TC_cnt, i - 1];  %사람의 위치 index저장
                TC_cnt = 0;
            end
        end
    end
end

if(TC_cnt ~= 0)
    if(TC_cnt < 15)  % 15미만의 점은 특이치로 판별하여 버림
        TC_matrix(i - TC_cnt : i - 1, :) = 0;
        TC_cnt = 0;
    elseif(TC_cnt > 45) % 45이상의 점은 특이치로 판별하여 버림
        TC_matrix(i - TC_cnt : i - 1, :) = 0;
        TC_cnt = 0;
    else % 15이상 45이하 사람으로 판별
        Human_cnt = Human_cnt + 1;  %사람의 수 카운트
        Distance(Human_cnt, 2) = 0;
        Distance(Human_cnt, :) = [i - TC_cnt, i - 1];  %사람의 위치 index저장
        TC_cnt = 0;
    end
end

%% Image
% 이미지에서 검출된 사람의 수만큼 각각 잘려진 이미지
for i = 1 : Human_cnt
    figure
    [Max_sub(i,1), Max_sub_Index(i,1)] = max(SD(Distance(i,1) :Distance(i,2),:));
    Max_sub_Index(i,1) = Max_sub_Index(i,1) + Distance(i,1);
    plot(rawdata(Max_sub_Index(i,1),:)) ;
    
    if(size(rawdata,1) < Max_sub_Index(i,1) + 15)
         Distance(i,1) = Max_sub_Index(i,1) - 15;
        Distance(i,2) = size(rawdata,1);
    elseif(Max_sub_Index(i,1) - 15 < 1)
        Distance(i,1) = 1;
        Distance(i,2) = Max_sub_Index(i,1) + 15;
    else
        Distance(i,1) = Max_sub_Index(i,1) - 15;
        Distance(i,2) = Max_sub_Index(i,1) + 15;
    end

    figure
    subplot(4,1,1),image(rawdata(Distance(i,1) :Distance(i,2),:),'CDataMapping','scaled');
    subplot(4,1,2), image(rawdata(Distance(i,1) :Distance(i,2),:),'CDataMapping','scaled');

    im = rawdata(Distance(i,1) :Distance(i,2),:);
    
    % L0-norm gradient minimization
    L = 2e-2;
    [u,ux,uy] = l0_grad_minimization(im,L);
    adu = mean(abs(ux).^2+abs(uy).^2,3);

    subplot(4,1,3);
    image(u,'CDataMapping','scaled');
    
    subplot(4,1,4);
    imshow(mat2gray(u));
end