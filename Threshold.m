%% Baseline threshold
SD = [];

SD = std(rawdata, 0, 2); %%거리에 대한 표준편차 배열

[Max, Index] = max(SD(50:end,:)); %% Max : 가장 큰 표준편차 값, Index : Max의 위치 값 앞쪽 50cm 이내는 제거
Index = Index + 49;
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

TC_matrix = SD > Dynamic_threshold;

TC_cnt = 0;
Human_cnt = 0;

for i = 1 : size(rawdata,1) % 행 길이 만큼 반복
    if (TC_matrix(i))
        TC_cnt = TC_cnt + 1;
    else
        if(TC_cnt > 0)
            if(TC_cnt < 4)
                TC_matrix(i - TC_cnt : i - 1, :) = 0;
                TC_cnt = 0;
            else
                Human_cnt = Human_cnt + 1;
                TC_cnt = 0;
            end
        end
    end
end

for i = 1 : size(rawdata,2) % 열 길이 만큼 반복
    Data(:,i) = rawdata(:,i) .* TC_matrix;
end
