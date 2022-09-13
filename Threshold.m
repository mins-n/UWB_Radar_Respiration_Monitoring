
Deviation = [];



%% Baseline threshold
Deviation = std(rawdata, 0, 2); %%거리에 대한 표준편차 배열

[Max, Index] = max(Deviation); %% Max : 가장 큰 표준편차 값, Index : Max의 위치 값
Pm = mean(Deviation(Index-1 : Index+1));
d0 = Index;
n = mean(Deviation);
Baseline_threshold = (Pm - n)/(2*d0 + 1) + n;

%% Dynamic threshold
di = 1 : size(rawdata,1);
Dynamic_threshold = d0^2 ./ di'.^2 .* Baseline_threshold;

%% Threshold crossing
TC_matrix = Deviation > Dynamic_threshold;

TC_cnt = 0;
Human_cnt = 0;

for i = 0 : size(rawdata,1)
    if (TC_matrix(i) == 1)
        TC_cnt = TC_cnt + 1;
    elseif( TC_cnt > 0)
        if( TC_cnt < 4)
            TC_matrix(i - TC_cnt : i - 1) = 0;
            TC_cnt = 0;
        elseif( TC_cnt > 8)
            TC_cnt = 0;
        else
            Human_cnt = Human_cnt + 1;
        end
    end
end

