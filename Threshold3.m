Windowsize = 400; % 20s 
Window_sliding = 3; % 윈도우 어디부터 시작할지 ex) 1이면 0 ~ 400
Window_rawdata = [];

%% Baseline threshold
SD = [];
rm = 20;
Window_rawdata = rawdata( : , Windowsize*(Window_sliding-1)+1:Windowsize*Window_sliding);

SD = std(rawdata, 0, 2); %%거리에 대한 표준편차 배열

[Max, Index] = max(SD(rm:end,:)); %% Max : 가장 큰 표준편차 값, Index : Max의 위치 값 앞쪽 50cm 이내는 제거
Index = Index + rm;

Pm = mean(SD(Index-1 : Index+1));
d0 = Index * 0.6445;
n = mean(SD);
Baseline_threshold = (Pm - n)/(2*d0 + 1) + n;

%% Dynamic threshold
di = [];
k = [];

di = 1 : size(rawdata,1);
k = (di.*0.6445).^2 / d0^2
Dynamic_threshold = Baseline_threshold ./ k';

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
            if(TC_cnt < 20)  % 20미만의 점은 특이치로 판별하여 버림
                TC_matrix(i - TC_cnt : i - 1, :) = 0;
                TC_cnt = 0;
%             elseif(TC_cnt > 70) % 70이상의 점은 특이치로 판별하여 버림
%                 TC_matrix(i - TC_cnt : i - 1, :) = 0;
%                 TC_cnt = 0;
            else % 20이상 70이하 사람으로 판별
                Human_cnt = Human_cnt + 1;  %사람의 수 카운트
                Distance(Human_cnt, 2) = 0;
                Distance(Human_cnt, :) = [i - TC_cnt, i - 1];  %사람의 위치 index저장
                TC_cnt = 0;
            end
        end
    end
end

if(TC_cnt ~= 0)
    if(TC_cnt < 20)  % 20미만의 점은 특이치로 판별하여 버림
        TC_matrix(i - TC_cnt : i - 1, :) = 0;
        TC_cnt = 0;
%     elseif(TC_cnt > 70) % 70이상의 점은 특이치로 판별하여 버림
%         TC_matrix(i - TC_cnt : i - 1, :) = 0;
%         TC_cnt = 0;
    else % 20이상 70이하 사람으로 판별
        Human_cnt = Human_cnt + 1;  %사람의 수 카운트
        Distance(Human_cnt, 2) = 0;
        Distance(Human_cnt, :) = [i - TC_cnt, i - 1];  %사람의 위치 index저장
        TC_cnt = 0;
    end
end
Max_sub = zeros(Human_cnt,1);
Max_sub_Index = zeros(Human_cnt,1);

minimize_rawdata = [];
minimize_rawdata_map = [];
aaa = 1;
%% Image
% 이미지에서 검출된 사람의 수만큼 각각 잘려진 이미지
for i = 1 : Human_cnt
    figure
    [Max_sub(i,1), Max_sub_Index(i,1)] = max(SD(Distance(i,1) :Distance(i,2),:));
    Max_sub_Index(i,1) = Max_sub_Index(i,1) + Distance(i,1);
    for j = 1 : fix(size(rawdata,2)/17)
        minimize_rawdata(i,j) = mean(rawdata(Max_sub_Index(i,1),aaa:aaa+17));
        aaa = aaa+17
    end
    aaa = 1;
    index1 = 1;
    index2 = 1;
    peak_mean1 = 0;
    peak_mean2 = 0;
    for k = 1 : fix(size(minimize_rawdata,2)/2)
        if(minimize_rawdata(i, 2*k)<minimize_rawdata(i, 2*k-1) && minimize_rawdata(i, 2*k)<minimize_rawdata(i, 2*k+1))
           peak_mean1 = minimize_rawdata(i, 2*k);
           index1=2*k;
        end
        if(minimize_rawdata(i, 2*k)>minimize_rawdata(i, 2*k-1) && minimize_rawdata(i, 2*k)>minimize_rawdata(i, 2*k+1))
           peak_mean2 = minimize_rawdata(i, 2*k);
           index2=2*k;
        end
        if(peak_mean1~=0 && peak_mean2~=0)
            if(index1<index2)
                for m = 1 : index2 -index1
                    minimize_rawdata_map(i,m) = minimize_rawdata(i,index1+m)-mean(minimize_rawdata(i,index1:index2));
                end
            else
                for m = 1 : index1-index2
                    minimize_rawdata_map(i,m) = minimize_rawdata(i,index2+m)-mean(minimize_rawdata(i,index2:index1));
                end
            end
            peak_mean1=0;
            peak_mean2=0;
        end
    end
    plot(minimize_rawdata_map(i,:)) ;
    if(size(rawdata,2) < Max_sub_Index(i,1) + 15)
         Distance(i,1) = Max_sub_Index(i,1) - 15;
        Distance(i,2) = size(rawdata,2);
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
    
    im = rawdata(Distance(i,1):Distance(i,2),:);
    
    % L0-norm gradient minimization
    L = 2e-2;
    [u,ux,uy] = l0_grad_minimization(im,L);
    adu = mean(abs(ux).^2+abs(uy).^2,3);

    subplot(4,1,3);
    image(u,'CDataMapping','scaled');
    
    subplot(4,1,4);
    imshow(mat2gray(u));
end
