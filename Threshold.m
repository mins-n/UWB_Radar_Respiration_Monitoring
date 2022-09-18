%% Baseline threshold
SD = [];

SD = std(rawdata, 0, 2); %%거리에 대한 표준편차 배열

[Max, Index] = max(SD(45:end,:)); %% Max : 가장 큰 표준편차 값, Index : Max의 위치 값 앞쪽 50cm 이내는 제거
Index = Index + 44;
Pm = mean(SD(Index-1 : Index+1));
d0 = Index;
n = mean(SD);
Baseline_threshold = (Pm - n)/(2*d0 + 1) + n;

%% Dynamic threshold
di = [];
k = [];

di = 1 : size(rawdata,1);
k = di.^2 / d0^2
Dynamic_threshold = Baseline_threshold ./ k';

%% Threshold crossing
TC_matrix = [];
Data = [];
Distance = [];


TC_matrix = SD > Dynamic_threshold; %SD가 threshold보다 큰 신호는 1, 아니면 0인 행렬

TC_cnt = 0; %threshold를 넘는 연속된 점 수
Human_cnt = 0;

for i = 1 : size(rawdata,1) % 행 길이 만큼 반복
    if (TC_matrix(i))
        TC_cnt = TC_cnt + 1; %점 수 세기
    else
        if(TC_cnt > 0)
            if(TC_cnt < 10)  % 10미만의 점은 특이치로 판별하여 버림
                TC_matrix(i - TC_cnt : i - 1, :) = 0;
                TC_cnt = 0;
            else % 10이상은 사람으로 판별
                Human_cnt = Human_cnt + 1;  %사람의 수 카운트
                Distance(Human_cnt, 2) = 0;
                Distance(Human_cnt, :) = [i - TC_cnt, i - 1];  %사람의 위치 index저장
                TC_cnt = 0;
            end
        end
    end
end

%필요없는 signal 0으로 만든 전체 이미지
for i = 1 : size(rawdata,2) % 열 길이 만큼 반복
    Data(:,i) = rawdata(:,i) .* TC_matrix; %Dynamic threshold 아래 부분 제거
end
subplot(Human_cnt+1,1,1), image(Data,'CDataMapping','scaled');

% 이미지에서 검출된 사람의 수만큼 각각 잘려진 이미지
for i = 1 : Human_cnt
    subplot(Human_cnt+1,1,i+1), image(rawdata(Distance(i,1) :Distance(i,2), :),'CDataMapping','scaled');
end
